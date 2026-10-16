import argparse
import multiprocessing as mp
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rl.base_assist_env import BaseAssistEnv as CurrentEnv
from rl.base_assist_env_v6 import BaseAssistEnv as V6Env
from rl.evaluate import V5Env


ENVIRONMENTS = {
    "v5": V5Env,
    "v6": V6Env,
    "v7": CurrentEnv,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train TidyBot++ SAC base-assistance policies"
    )

    parser.add_argument(
        "--version",
        choices=ENVIRONMENTS.keys(),
        default="v7",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25000,
        help="Additional training timesteps",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="New checkpoint name, e.g. v7_50k",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Existing checkpoint to continue from",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    models_dir = Path("rl/models")
    logs_dir = Path("rl/logs")
    tensorboard_dir = logs_dir / "tensorboard"

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    output = models_dir / f"sac_base_assist_{args.name}"
    model_file = Path(f"{output}.zip")
    buffer_file = Path(f"{output}_buffer.pkl")
    monitor_file = logs_dir / f"sac_{args.name}_monitor.csv"

    existing = [
        p for p in (model_file, buffer_file, monitor_file)
        if p.exists()
    ]

    if existing and not args.overwrite:
        parser.error(
            "Output already exists: "
            + ", ".join(str(p) for p in existing)
        )

    if args.resume:
        source = models_dir / f"sac_base_assist_{args.resume}"
        source_model = Path(f"{source}.zip")
        source_buffer = Path(f"{source}_buffer.pkl")

        if not source_model.exists():
            parser.error(f"Checkpoint not found: {source_model}")

        if not source_buffer.exists():
            parser.error(
                f"Replay buffer not found: {source_buffer}"
            )

        if source == output:
            parser.error(
                "Use a different --name when resuming, "
                "so the original checkpoint is preserved."
            )

    env = ENVIRONMENTS[args.version]()
    env.unwrapped.randomize_reach = True

    monitored_env = Monitor(
        env,
        filename=str(monitor_file),
    )

    try:
        if args.resume:
            model = SAC.load(
                str(source),
                env=monitored_env,
                device="cpu",
            )
            model.load_replay_buffer(str(source_buffer))
            print(f"Resuming from: {source_model}")
        else:
            model = SAC(
                "MlpPolicy",
                monitored_env,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=5000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                tensorboard_log=str(tensorboard_dir),
                device="cpu",
                seed=42,
                verbose=1,
            )

        print(
            f"\nEnvironment: {args.version}"
            f"\nCheckpoint: {args.name}"
            f"\nTraining steps: {args.steps:,}\n"
        )

        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=not bool(args.resume),
            log_interval=10,
            progress_bar=True,
        )

        model.save(str(output))
        model.save_replay_buffer(str(output) + "_buffer")

        print(f"\nTraining complete!")
        print(f"Model: {model_file}")
        print(f"Replay buffer: {buffer_file}")

    finally:
        monitored_env.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()