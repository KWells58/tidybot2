import argparse
import time
from itertools import count
from mujoco_env import MujocoEnv
from constants import POLICY_CONTROL_PERIOD
from policies import TeleopPolicy

def run_episode(env, policy):
    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    # Wait for user to press "Start episode"
    print('Press "Start episode" in the web app when ready to start new episode')
    policy.reset()
    print('Starting new episode')

    episode_ended = False
    start_time = time.time()
    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Get latest observation
        obs = env.get_obs()

        # Get action
        action = policy.step(obs)

        # No action if teleop not enabled
        if action is None:
            continue

        # Execute valid action on robot
        if isinstance(action, dict):
            env.step(action)

        # Episode ended
        elif not episode_ended and action == 'end_episode':
            episode_ended = True
            print('Episode ended')

            print('Teleop is now active. Press "Reset env" in the web app when ready to proceed.')

        # Ready for env reset
        elif action == 'reset_env':
            break

def main(args):
    env = MujocoEnv()
    policy = TeleopPolicy()
    try:
        while True:
            run_episode(env, policy)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
