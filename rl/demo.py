import argparse
import multiprocessing as mp
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

from rl.base_assist_env_v6 import BaseAssistEnv


SCENE = "models/stanford_tidybot/scene.xml"


def record_episode(model, reach, assisted):
    env = BaseAssistEnv()
    env.randomize_reach = False
    env.reach_distance = reach

    frames = []
    errors = []
    sigmas = []

    try:
        obs, _ = env.reset()

        for step in range(env.max_steps):
            if assisted:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            state = env.sim.get_obs()

            frames.append({
                "base_pose": state["base_pose"].copy(),
                "arm_q": state["arm_q"].copy(),
            })

            errors.append(float(info["tracking_error"]))
            sigmas.append(float(info["sigma_min"]))

            if terminated or truncated:
                break

    finally:
        env.close()

    return {
        "frames": frames,
        "mean_error": float(np.mean(errors)),
        "min_sigma": float(np.min(sigmas)),
        "unsafe_fraction": float(np.mean(np.array(sigmas) < 0.08)),
    }


def find_joint(model, names):
    for name in names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id >= 0:
            return model.jnt_qposadr[joint_id]
    raise ValueError(f"Could not find joint: {names}")


def replay_episode(recording, title, fps):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    # These names must match the joints in the MJCF model.
    base_x = find_joint(model, ["joint_x"])
    base_y = find_joint(model, ["joint_y"])
    base_yaw = find_joint(model, ["joint_th"])

    arm_joint_addresses = [
        find_joint(model, [
            f"joint_{i}",
            f"joint{i}",
            f"arm_joint_{i}",
        ])
        for i in range(1, 8)
    ]

    print(f"\nOpening MuJoCo viewer: {title}")
    print("Close the viewer window to continue.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for frame in recording["frames"]:
            if not viewer.is_running():
                break

            base = frame["base_pose"]
            arm = frame["arm_q"]

            data.qpos[base_x] = base[0]
            data.qpos[base_y] = base[1]
            data.qpos[base_yaw] = base[2]

            for address, value in zip(arm_joint_addresses, arm):
                data.qpos[address] = value

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / fps)

        # Keep the final configuration visible.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(
        description="Replay SAC base-assistance episodes in MuJoCo"
    )
    parser.add_argument("--model", default="v6_50k")
    parser.add_argument("--reach", type=float, default=0.60)
    parser.add_argument("--fps", type=float, default=20.0)

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    if not args.model.startswith("v6"):
        parser.error("This demo currently supports V6 models only.")

    model = SAC.load(f"rl/models/sac_base_assist_{args.model}")

    print("\nRecording no-assistance episode...")
    baseline = record_episode(model, args.reach, assisted=False)

    print("Recording SAC-assisted episode...")
    assisted = record_episode(model, args.reach, assisted=True)

    for label, recording in [
        ("No assistance", baseline),
        ("SAC assistance", assisted),
    ]:
        print(
            f"{label:16s} | "
            f"Mean error: {recording['mean_error'] * 1000:.2f} mm | "
            f"Min sigma: {recording['min_sigma']:.4f} | "
            f"Unsafe: {recording['unsafe_fraction'] * 100:.1f}%"
        )

    replay_episode(baseline, "No assistance", args.fps)
    replay_episode(assisted, "SAC assistance", args.fps)


if __name__ == "__main__":
    mp.freeze_support()
    main()