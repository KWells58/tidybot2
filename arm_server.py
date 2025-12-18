# Authors: Yihuai Gao, Jimmy Wu
# Date: October 2024
#
# This RPC server allows other processes to communicate with the ARX5 arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

import math
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import POLICY_CONTROL_PERIOD
from ik_solver import IKSolver

import sys
sys.path.append('arx5-sdk/python')
import arx5_interface as arx5

class Arm:
    def __init__(self):
        robot_config = arx5.RobotConfigFactory.get_instance().get_config('L5')
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config('joint_controller', robot_config.joint_dof)
        controller_config.default_preview_time = POLICY_CONTROL_PERIOD  # Enable interpolation

        # Joint limits
        joint_pos_max = robot_config.joint_pos_max.copy()
        joint_pos_max[1] = math.radians(210)     # Enable arm to reach floor
        joint_pos_max[2] = math.radians(166.24)  # Prevent arm hyperextension
        robot_config.joint_pos_max = joint_pos_max

        # Gripper limits
        robot_config.gripper_vel_max = 0.5
        robot_config.gripper_torque_max = 5.5

        # Arm controller and IK solver
        self.arm = arx5.Arx5JointController(robot_config, controller_config, 'can1')  # Use can1 since can0 is used by mobile base
        self.joint_cmd = arx5.JointState(robot_config.joint_dof)
        self.ik_solver = IKSolver(ee_offset=0.155)  # 0.155 from arx5-sdk/models/L5.urdf

        # Use home position x-value as workspace boundary
        self.home_pos_x = self.arm.get_home_pose()[0]

    def reset(self):
        self.arm.reset_to_home()

    def execute_action(self, action):
        # Apply workspace boundary
        target_pos = action['arm_pos']
        if target_pos[0] < self.home_pos_x:
            target_pos[0] = self.home_pos_x

        joint_state = self.arm.get_joint_state()
        qpos = self.ik_solver.solve(target_pos, action['arm_quat'], joint_state.pos())
        self.joint_cmd.pos()[:] = qpos
        self.joint_cmd.gripper_pos = 0.088 * (1 - action['gripper_pos'].item())  # 0.088 from arx5-sdk/include/app/config.h
        self.arm.set_joint_cmd(self.joint_cmd)

    def get_state(self):
        eef_state = self.arm.get_eef_state()
        pose_6d = eef_state.pose_6d()
        arm_quat = R.from_euler('xyz', pose_6d[3:]).as_quat()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        state = {
            'arm_pos': pose_6d[:3],
            'arm_quat': arm_quat,
            'gripper_pos': np.array([(0.088 - eef_state.gripper_pos) / 0.088]),
        }
        return state

    def close(self):
        self.arm.reset_to_home()

class ArmManager(MPBaseManager):
    pass

ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}')
    server.serve_forever()
    # import time
    # import numpy as np
    # manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # manager.connect()
    # arm = manager.Arm()
    # try:
    #     arm.reset()
    #     for i in range(50):
    #         arm.execute_action({
    #             'arm_pos': np.array([0.2527, 0.0, 0.1635]),
    #             'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
    #             'gripper_pos': np.zeros(1),
    #         })
    #         print(arm.get_state())
    #         time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    # finally:
    #     arm.close()
