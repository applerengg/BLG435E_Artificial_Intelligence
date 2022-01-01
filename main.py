#os.environ["SDL_VIDEODRIVER"] = "dummy"

from collections import namedtuple

import numpy as np

from snake import CustomSnake as Snake
from snake_environment import SnakeMode

from DQN import DQN

import time
from torch.utils.tensorboard import SummaryWriter

# Are we rendering or not
# RENDER = False
RENDER = False

# Number of steps to train
# NUM_STEPS = 200
NUM_STEPS = 50_000

# The step limit per episode, since we do not want infinite loops inside episodes
MAX_STEPS_PER_EPISODE = 200

# Here, we are creating a data structure to store our transitions
# This is just a convenient way to store
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

gamemode = SnakeMode.NOTAIL
# gamemode = SnakeMode.CLASSIC
# gamemode = SnakeMode.TRON
# Here, we are creating the environment with our predefined observation space
env = Snake(mode=gamemode, render=RENDER)

# Observation and action space
obs_space = env.observation_space
number_of_states = env.observation_space.shape[0]

action_space = env.action_space
number_of_actions = env.action_space.n
print("The observation space: {}".format(obs_space))
# Output: The observation space: Box(n,)
print("The action space: {}".format(action_space))
# Output: The action space: Discrete(m)

HYPERPARAMETERS = {
    "number_of_states": number_of_states,
    "number_of_actions": number_of_actions,
    "number_of_steps": NUM_STEPS,
    # "replay_buffer_capacity": 10000,
    "replay_buffer_capacity": 5000,
    # "learning_rate": 1e-2,
    "learning_rate": 0.5,
    # "batch_size": 256,
    "batch_size": 500,
    "gamma": 0.995,
    # "epsilon_annealing": "linear",
    "epsilon_annealing": "exponential",
    "decay": 0.99990,
}


def main():

    agent = DQN(HYPERPARAMETERS)

    episodes = 0
    steps = 0
    episode_rewards = []
    episode_steps = []
    episode_apples = []
    
    start_time = time.perf_counter()

    comment_string = str(gamemode) \
                        + "_" + str(NUM_STEPS) \
                        + "_LR" + str(HYPERPARAMETERS["learning_rate"]) \
                        + "_" + HYPERPARAMETERS["epsilon_annealing"]
    if HYPERPARAMETERS["epsilon_annealing"] == "exponential":
        comment_string += "-" + str(HYPERPARAMETERS["decay"])
    
    # comment_string += "_snake-dirs"
        
    writer = SummaryWriter(comment= "_" + comment_string)

    while steps < NUM_STEPS:
        state = env.reset()
        episode_reward = 0
        while True:
            steps += 1
            # The select action method inside DQN will select based on policy or random, depending on the epsilon value
            action = agent.select_action(state)

            # Here, we will step the environment with the action
            # Next_state: the state after the action is taken
            # Reward: The reward agent will get. It is generally
            # 1 if the agent wins the game, -1 if the agent loses, 0 otherwise
            # You can add intermediate rewards other than win-lose conditions
            # Done: is the game finished
            # Info: Further info you can get from the environment, you can ignore this part
            next_state, reward, done, info = env.step(action)

            # Render each frame?
            if RENDER:
                env.render()

            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)

            episode_reward += reward
            # print(f"{reward = } | {episode_reward = } | {steps = } ")

            if steps % 500 == 0:
                if episodes > 10:
                    print("Step: {}, Epsilon: {:.5f}, Mean Reward for last 10: {:.2f}".format(
                        steps, agent.e, np.average(episode_rewards[:-10])
                        ))


            if done or env.elapsed_steps >= MAX_STEPS_PER_EPISODE:
                episodes += 1
                episode_rewards.append(episode_reward)
                episode_steps.append(env.elapsed_steps)
                episode_apples.append(env.apple_count)
                if agent.update(): # if updated (memory is full)
                    writer.add_scalar(f"Reward - mean of last 100 Eps", np.mean(episode_rewards[:-100]), steps)
                    writer.add_scalar("Steps - mean of last 100 Eps", np.mean(episode_steps[:-100]), steps)
                    writer.add_scalar("Apples Eaten - mean of last 100", np.mean(episode_apples[:-100]), steps)
                    writer.add_scalar("Loss", agent.loss, steps)
                # writer.add_scalar("epsilon", agent.e, steps)
                # writer.add_scalar("reward_episode", episode_reward, steps)
                # We do not want the env to run indefinitely
                # When a done condition is met, we finish
                # We want to update the DQN
                break

            
            state = next_state

    writer.flush()
    writer.close()
    # agent.writer.close()

    # agent.save(str(gamemode))
    agent.save(comment_string)

    # Delete the current pygame instance
    env.quit()

    # print("\n", episode_rewards)
    print(f"\n{steps} steps, {len(episode_rewards)} episodes, "
        + f"max episode reward: {max(episode_rewards)}, "
        + f"max steps in an episode: {max(episode_steps)} ")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\n > time: {elapsed:.2f} sec. ({elapsed / 60 :.3f} min.)")

    # return
    # Create a new pygame instance with render enabled
    # render_env = Snake(mode=SnakeMode.NOTAIL, render=True)
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
            time.sleep(0.05)
            if done:
                print(f"Done. Reward: {render_env.total_reward: >6}"
                    + f"  Apples Eaten: {render_env.apple_count: >2}"
                    + f"  Steps: {render_env.elapsed_steps: >4} ")
                test_episode_rewards.append(render_env.total_reward)
                test_apples.append(render_env.apple_count)
                test_episode_steps.append(render_env.elapsed_steps)
                time.sleep(0.5)
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


