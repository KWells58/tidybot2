import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from rl.base_assist_env import BaseAssistEnv as CurrentEnv
from rl.base_assist_env_v5 import BaseAssistEnv as OriginalV5Env
from rl.base_assist_env_v6 import BaseAssistEnv as V6Env


class V5Env(gym.ObservationWrapper):
    """Adapt 23-value observations to the original V5 format."""

    def __init__(self):
        super().__init__(OriginalV5Env())

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation[:22], dtype=np.float32)


DEFAULT_REACHES = [0.20, 0.40, 0.50, 0.60]
SIGMA_THRESHOLD = 0.08


def environment_for_model(model_name):
    if model_name.startswith("v5"):
        return V5Env
    if model_name.startswith("v6"):
        return V6Env
    if model_name.startswith("v7"):
        return CurrentEnv

    raise ValueError(f"Unsupported model version: {model_name}")


def model_path(model_name):
    return Path("rl/models") / f"sac_base_assist_{model_name}"


def run_episode(env, model, distance, use_sac):
    base_env = env.unwrapped
    base_env.randomize_reach = False
    base_env.reach_distance = distance

    obs, info = env.reset()

    rewards = []
    errors = []
    sigmas = []
    actions = []
    common_costs = []

    previous_action = np.zeros(2, dtype=np.float32)

    for step in range(base_env.max_steps):
        if use_sac:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(2, dtype=np.float32)

        action = np.asarray(action, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        tracking_error = float(info["tracking_error"])
        sigma = float(info["sigma_min"])

        # Apply identical scoring coefficients to all models.
        common_cost = (
            10.0 * tracking_error
            + 20.0 * max(0.0, SIGMA_THRESHOLD - sigma)
            + 0.2 * np.linalg.norm(action)
            + 0.05 * np.linalg.norm(action - previous_action)
        )

        rewards.append(float(reward))
        errors.append(tracking_error)
        sigmas.append(sigma)
        actions.append(action.copy())
        common_costs.append(float(common_cost))

        previous_action = action.copy()

        if terminated or truncated:
            break

    actions = np.asarray(actions)
    sigmas = np.asarray(sigmas)

    return {
        "reach_m": distance,
        "reward": float(np.sum(rewards)),
        "common_reward": -float(np.sum(common_costs)),
        "mean_error_m": float(np.mean(errors)),
        "min_sigma": float(np.min(sigmas)),
        "final_sigma": float(sigmas[-1]),
        "base_x_m": float(info["base_pose"][0]),
        "base_y_m": float(info["base_pose"][1]),
        "mean_action_x": float(np.mean(actions[:, 0])),
        "unsafe_fraction": float(
            np.mean(sigmas < SIGMA_THRESHOLD)
        ),
    }


def evaluate_model(model_name, reaches):
    env_class = environment_for_model(model_name)
    env = env_class()

    try:
        model = SAC.load(str(model_path(model_name)))

        if model.observation_space.shape != env.observation_space.shape:
            raise ValueError(
                f"Observation mismatch for {model_name}: "
                f"model={model.observation_space.shape}, "
                f"env={env.observation_space.shape}"
            )

        results = []

        for reach in reaches:
            for label, use_sac in [
                ("No assistance", False),
                (model_name, True),
            ]:
                result = run_episode(
                    env, model, reach, use_sac
                )

                result["controller"] = label
                result["environment"] = model_name[:2]
                results.append(result)

                print(
                    f"{label:16s} | "
                    f"Reach: {reach:.2f} m | "
                    f"Common reward: {result['common_reward']:8.3f} | "
                    f"Error: {result['mean_error_m'] * 1000:6.2f} mm | "
                    f"Min sigma: {result['min_sigma']:.4f} | "
                    f"Base X: {result['base_x_m']:+.3f} m | "
                    f"Base Y: {result['base_y_m']:+.3f} m | "
                    f"Unsafe: {result['unsafe_fraction'] * 100:.1f}%"
                )

        return results

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TidyBot++ base-assistance policies"
    )

    parser.add_argument(
        "--model",
        default="v6_50k",
        help="Model checkpoint identifier, e.g. v6_50k",
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple checkpoints",
    )

    parser.add_argument(
        "--reach",
        type=float,
        nargs="+",
        default=DEFAULT_REACHES,
        help="Reach distances in meters",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path",
    )

    args = parser.parse_args()
    models = args.compare if args.compare else [args.model]

    all_results = []

    for model_name in models:
        print(f"\n===== MODEL {model_name} =====")

        all_results.extend(
            evaluate_model(model_name, args.reach)
        )

    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(all_results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nSaved results to {output}")


if __name__ == "__main__":
    mp.freeze_support()
    main()