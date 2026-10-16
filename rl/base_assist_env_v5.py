from scipy.spatial.transform import Rotation
from curses import raw
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mujoco_env import MujocoEnv
from rl.singularity import SingularityMonitor


class BaseAssistEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.sim = MujocoEnv(
            render_images=False,
            show_viewer=False,
            sync_mode=True
        )

        self.monitor = SingularityMonitor()

        # RL chooses small X/Y base movements.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # 7 joints + 3 arm position + 3 base pose
        # + 3 desired arm position + 1 sigma
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )

        self.target_arm_pos = None
        self.target_arm_quat = None
        self.target_base_pose = np.zeros(3)
        self.previous_action = np.zeros(2)
        self.step_count = 0
        self.max_steps = 100
        self.reach_distance = 0.20
        self.randomize_reach = False

    def _observe(self):
        raw = self.sim.get_obs()
        theta = raw["base_pose"][2]
        c, s = np.cos(theta), np.sin(theta)

        R_inv = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1],
        ])

        current_local_target = R_inv @ (
            self.target_world_pos
            - np.array([*raw["base_pose"][:2], 0.0])
        )

        metrics = self.monitor.measure(
            raw["arm_q"],
            raw["base_pose"]
        )

        target_velocity_world = np.array([
            self.reach_distance / (self.max_steps * 50 * self.monitor.model.opt.timestep),
            0.0,
            0.0,
        ])

        target_velocity_local = R_inv @ target_velocity_world
        remaining_reach = self.reach_distance * (
            1.0 - self.step_count / self.max_steps
        )
        observation = np.concatenate([
            raw["arm_q"],
            raw["arm_pos"],
            raw["base_pose"],
            current_local_target,
            target_velocity_local,
            [metrics["sigma_min"]],
            self.previous_action,
            [remaining_reach],
        ]).astype(np.float32)

        return observation, raw, metrics

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_reach:
            self.reach_distance = float(
                self.np_random.uniform(0.10, 0.60)
            )

        self.sim.reset()

        raw = self.sim.get_obs()

        self.target_arm_pos = raw["arm_pos"].copy()
        self.target_arm_quat = raw["arm_quat"].copy()
        self.target_base_pose = raw["base_pose"].copy()
        # Save the desired end-effector orientation
        # in world coordinates.
        base_yaw = raw["base_pose"][2]

        self.target_world_rot = (
            Rotation.from_euler("z", base_yaw)
            * Rotation.from_quat(self.target_arm_quat)
        )

        theta = raw["base_pose"][2]
        c, s = np.cos(theta), np.sin(theta)

        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])

        self.target_world_pos = (
            R @ self.target_arm_pos
            + np.array([*raw["base_pose"][:2], 0.0])
        )

        self.previous_action[:] = 0
        self.step_count = 0
        self.initial_world_pos = self.target_world_pos.copy()

        obs, _, metrics = self._observe()

        return obs, {
            "sigma_min": metrics["sigma_min"],
            "reach_distance": float(self.reach_distance),
        }

    def step(self, action):
        # Simulate an operator reaching forward over the episode.
        reach_distance = self.reach_distance
        reach_duration = 100

        progress = min(
            (self.step_count + 1) / reach_duration,
            1.0
        )

        self.target_world_pos = self.initial_world_pos.copy()
        self.target_world_pos[0] += reach_distance * progress

        action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0
        )

        # Maximum commanded base movement per step: 1 cm.
        base_delta = action * 0.01

        self.target_base_pose[:2] += base_delta

        # Convert the fixed world target into the
        # base's current local coordinate frame.
        current_base = self.sim.get_obs()["base_pose"]

        theta = current_base[2]
        c, s = np.cos(theta), np.sin(theta)

        R_inv = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1],
        ])

        local_target = R_inv @ (
            self.target_world_pos
            - np.array([*current_base[:2], 0.0])
        )

        # Convert desired world orientation into
        # the current base's local frame.
        local_rotation = (
            Rotation.from_euler("z", -theta)
            * self.target_world_rot
        )

        local_quat = local_rotation.as_quat()

        # Send the arm target and base command together.
        self.sim.step({
            "base_pose": self.target_base_pose.copy(),
            "arm_pos": local_target,
            "arm_quat": local_quat,
        }, num_steps=50)

        _, raw, metrics = self._observe()

        theta = raw["base_pose"][2]
        c, s = np.cos(theta), np.sin(theta)

        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])

        actual_world_pos = (
        R @ raw["arm_pos"]
    + np.array([*raw["base_pose"][:2], 0.0])
        )

        tracking_error = np.linalg.norm(
            actual_world_pos - self.target_world_pos
        )

        sigma = metrics["sigma_min"]

        # Initial experimental reward weights.
        tracking_penalty = 10.0 * tracking_error
        sigma_threshold = 0.08
        singularity_penalty = 20.0 * max(0.0, sigma_threshold - sigma)
        movement_penalty = 0.2 * np.linalg.norm(action)
        smoothness_penalty = 0.05 * np.linalg.norm(
            action - self.previous_action
        )

        reward = -(
            tracking_penalty
            + singularity_penalty
            + movement_penalty
            + smoothness_penalty
        )

        self.previous_action = action.copy()
        self.step_count += 1

        obs, _, _ = self._observe()

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "sigma_min": sigma,
            "tracking_error": float(tracking_error),
            "base_pose": raw["base_pose"].copy(),
            "reach_distance": float(self.reach_distance),
            "tracking_penalty": float(tracking_penalty),
            "singularity_penalty": float(singularity_penalty),
            "movement_penalty": float(movement_penalty),
            "smoothness_penalty": float(smoothness_penalty),
        }

        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()
