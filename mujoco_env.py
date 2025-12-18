import multiprocessing as mp
import time
from multiprocessing import shared_memory
import mujoco
import mujoco.viewer
import numpy as np

import sys
sys.path.append('..')
from constants import POLICY_CONTROL_PERIOD
from ik_solver import IKSolver
# from ik_solver_mink import IKSolver

class ShmState:
    def __init__(self, existing_instance=None):
        arr = np.empty(3 + 4 + 1)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.arm_pos = self.data[:3]
        self.arm_quat = self.data[3:7]
        self.initialized = self.data[7:8]
        self.initialized[:] = 0.0

    def close(self):
        self.shm.close()

class ArmController:
    def __init__(self, qpos, qvel, ctrl):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

        # IK solver
        self.ik_solver = IKSolver(ee_offset=0.145)

    def reset(self):
        # Initialize arm in "retract" configuration
        self.qpos[:] = np.zeros(6)
        self.ctrl[:] = self.qpos

    def control_callback(self, command):
        if command is not None:
            if 'arm_pos' in command:
                # Run inverse kinematics on new target pose
                qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat'], self.qpos)
                qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi  # Unwrapped joint angles

                # Set target arm qpos
                self.ctrl[:] = qpos

class MujocoSim:
    def __init__(self, mjcf_path, command_queue, shm_state, show_viewer=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        # Enable gravity compensation for everything except objects
        self.model.body_gravcomp[:] = 1.0

        # Cache references to array slices
        arm_dofs = 7
        qpos_arm = self.data.qpos[:arm_dofs]
        qvel_arm = self.data.qvel[:arm_dofs]
        ctrl_arm = self.data.ctrl[:arm_dofs]

        # Controllers
        self.arm_controller = ArmController(qpos_arm, qvel_arm, ctrl_arm)

        # Shared memory state for observations
        self.shm_state = ShmState(existing_instance=shm_state)

        # Variables for calculating arm pos and quat
        site_id = self.model.site('grasp_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)

        # Reset the environment
        self.reset()

        # Set control callback
        mujoco.set_mjcb_control(self.control_callback)

    def reset(self):
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset controllers
        self.arm_controller.reset()

    def control_callback(self, *_):
        # Check for new command
        command = None if self.command_queue.empty() else self.command_queue.get()
        if command == 'reset':
            self.reset()

        # Control callbacks
        self.arm_controller.control_callback(command)

        # Update arm pos
        self.shm_state.arm_pos[:] = self.site_xpos

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        self.shm_state.arm_quat[:] = self.site_quat

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

    def launch(self):
        if self.show_viewer:
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)

        else:
            # Run headless simulation at real-time speed
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)

class MujocoEnv:
    def __init__(self):
        self.mjcf_path = 'models/arx5/scene_pd.xml'
        self.command_queue = mp.Queue(1)

        # Shared memory for state observations
        self.shm_state = ShmState()

        # Start physics loop
        mp.Process(target=self.physics_loop, daemon=True).start()

    def physics_loop(self):
        # Create sim
        sim = MujocoSim(self.mjcf_path, self.command_queue, self.shm_state)

        # Launch sim
        sim.launch()  # Launch in same thread as creation to avoid segfault

    def reset(self):
        self.shm_state.initialized[:] = 0.0
        self.command_queue.put('reset')

        # Wait for state publishing to initialize
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

    def get_obs(self):
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        obs = {
            'base_pose': np.zeros(3),
            'arm_pos': self.shm_state.arm_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': np.zeros(1),
        }
        return obs

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.command_queue.put(action)

    def close(self):
        self.shm_state.close()
        self.shm_state.shm.unlink()

if __name__ == '__main__':
    env = MujocoEnv()
    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    'arm_pos': 0.1 * np.random.rand(3) + np.array([0.2, 0.0, 0.35]),
                    'arm_quat': np.random.rand(4),
                }
                env.step(action)
                obs = env.get_obs()
                print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
