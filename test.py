from collections import namedtuple

import numpy as np
import torch

from snake import CustomSnake as Snake
from snake_environment import SnakeMode

from DQN import DQN

import time
from torch.utils.tensorboard import SummaryWriter



HYPERPARAMETERS = {
    "number_of_states": 11,
    "number_of_actions": 4,
    "number_of_steps": 50_000,
    # "replay_buffer_capacity": 10000,
    "replay_buffer_capacity": 5000,
    # "learning_rate": 1e-2,
    "learning_rate": 1e-2,
    # "batch_size": 256,
    "batch_size": 500,
    "gamma": 0.995,
    # "epsilon_annealing": "linear",
    "epsilon_annealing": "exponential",
    "decay": 0.99990,
}


def epsilon_func():
    while 1:
        yield 0.0

def main():
    gamemode = SnakeMode.TRON
    agent = DQN(HYPERPARAMETERS)
    agent.epsilon = epsilon_func()
    agent.act_net.load_state_dict(
        # torch.load("SnakeMode.CLASSIC_50000_exponential-0.9999/target_Q.pt")
        torch.load("SnakeMode.TRON_40000_LR0.01_exponential-0.9999/target_Q.pt")
        # torch.load("SnakeMode.NOTAIL_50000_LR0.01_exponential-0.9999/target_Q.pt")
    )
    render_env = Snake(mode=gamemode, render=True)
    state = render_env.reset()
    test_episode_rewards = []
    test_apples = []
    test_episode_steps = []
    while True:
        try:
            action = agent.select_action(state)
            next_state, reward, done, info = render_env.step(action)
            render_env.render()
            # time.sleep(0.05)
            if done:
                print(f"Done. Reward: {render_env.total_reward: >6}"
                    + f"  Apples Eaten: {render_env.apple_count: >2}"
                    + f"  Steps: {render_env.elapsed_steps: >4} "
                    + f"  epsilon: {agent.e}")
                # if render_env.apple_count >= 10:
                if render_env.elapsed_steps >= 10:
                    a = input()  # wait
                test_episode_rewards.append(render_env.total_reward)
                test_apples.append(render_env.apple_count)
                test_episode_steps.append(render_env.elapsed_steps)
                # time.sleep(0.5)
                state = render_env.reset()
            else:
                state = next_state
        except KeyboardInterrupt:
            break
    
    print(
        f"\n > {len(test_apples)} Test Games: "
      + f"\n   max apples: {np.max(test_apples): >5}  "
      + f"max reward: {np.max(test_episode_rewards): >8}  "
      + f"max steps: {np.max(test_episode_steps): >5}  "
      + f"\n   avg apples: {np.mean(test_apples): >5.2f}  "
      + f"avg reward: {np.mean(test_episode_rewards): >8.3f}  "
      + f"avg steps: {np.mean(test_episode_steps): >5.1f}  "
    )


if __name__ == '__main__':
    main()