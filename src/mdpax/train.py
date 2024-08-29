#!/home/titan/workspace/mlos/venvs/jax/bin/python3
from environment import *
from dqn_agent import Agent
import jax
from jax import jit, random, numpy as jnp
from gymnasium import spaces
import os
import time

import wandb


# ENV LOGIC (REWARD AND TRANSITION FUNCTIONS)
# both will be jitted
def example_reward_function(state, target_state):
    """Define your reward logic here."""
    is_goal = jnp.all(jnp.abs(state - target_state) <= 0.1)
    current_distance = jnp.linalg.norm(state - target_state)
    total_distance = jnp.sqrt(2)
    percentage = current_distance / total_distance
    distance_reward = jnp.where(percentage < 0.33, -1, jnp.where(percentage < 0.66, -2, -3))
    reward = jnp.where(is_goal, 15.0, distance_reward)
    return reward


def example_transition_function(state, action, state_space_shape, _key):
    """Define your state transition logic here."""
    x, y = state
    _key, subkey = random.split(_key)

    def magic(key):
        # return random.uniform(key, minval=-1, maxval=1, shape=(2,), dtype=dtype)
        return random.uniform(key, (2,), minval=-1, maxval=1, dtype=jnp.float32)

    def action_one(_):
        return x + y

    def action_two(_):
        return jnp.abs(x - y)

    def action_three(_):
        return x / y

    def action_four(_):
        return y / x

    fp_key = jax.lax.switch(action, [action_one, action_two, action_three, action_four], None)
    x, y = magic(subkey) * fp_key
    return jnp.array([x, y], dtype=jnp.float32)


# ENV SETTINGS


def setup_environment():

    dimensions = env_min, env_max = -jnp.inf, jnp.inf
    dtype = jnp.float32

    state_space = spaces.Box(low=env_min, high=max(env_min, env_max), shape=(1, len(dimensions)), dtype=dtype)

    action_space_n = 4
    action_space = spaces.Discrete(action_space_n)

    target_state = jnp.array([0.0, 0.0], dtype=dtype)
    initial_state = jnp.array([-1.0, 1.0], dtype=dtype)

    seed = 0o020304 #my birthday :D

    config = EnvironmentConfig(
        seed=seed,
        state_space=state_space,
        action_space=action_space,
        initial_state=initial_state,
        target_state=target_state,
        reward_function=jit(example_reward_function),
        transition_function=jit(example_transition_function),
    )
    return create_environment(config)


def save_agent(agent, output_dir, episode):
    filename = os.path.join(output_dir, f"agent_{episode}.pkl")
    agent.save(filename)
    print(f"\n\n {episode} Episodes: saved model to {filename}\n\n")


def load_latest_agent(agent, output_dir):
    files = [f for f in os.listdir(output_dir) if f.startswith("agent_") and f.endswith(".pkl")]
    if not files:
        print("No saved agents found.")
        return False

    latest_file = max(files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    latest_file_path = os.path.join(output_dir, latest_file)

    confirmation = input(f"Do you want to load the latest agent from {latest_file_path}? (y/n): ")
    if confirmation.lower() != "y":
        return False

    agent.load(latest_file_path)
    print(f"Agent loaded from {latest_file_path}")
    return True


def train_agent(agent, env, num_episodes, num_iterations, batch_size, output_dir):
    start_time = time.time()
    global_steps = 0
    global_rewards = []

    mean_episode_reward = 0

    wandb.init(
        project="moon",
        entity="t4r3k-carleton-university",
        config={
            "batch_size": batch_size,
            "num_episodes": num_episodes,
            "num_iterations": num_iterations,
            "learning_rate": agent.learning_rate,
            "epsilon_decay": agent.epsilon_decay,
            "gamma": agent.gamma,
            "epsilon_min": agent.epsilon_min,
        },
    )

    for episode in range(num_episodes):
        state, _, _ = env.reset()
        episode_reward = 0

        for t in range(num_iterations):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            global_steps += 1
            episode_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done or t == (num_iterations - 1):
                break
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        global_rewards.append(episode_reward)
        mean_episode_reward = jnp.mean(jnp.array(global_rewards)) if global_rewards else 0.0
        # make mean_episode_reward a scalar
        if len(agent.memory) > batch_size:
            loss = agent.replay(batch_size)
        else:
            loss = 0.0

        print(
            f"Time: {int(hours):02}h {int(minutes):02}m {int(seconds):02}s, Episode: {episode}, Steps: {global_steps}, epr: {float(mean_episode_reward):.3}, Loss: {float(loss):.3} Score: {episode_reward}, Done: {done}"
        )
        wandb.log(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "loss": float(loss),
                "mean_episode_reward": float(mean_episode_reward),
                "global_steps": global_steps,
            }
        )

        if episode % 1000 == 0:
            save_agent(agent, output_dir, episode)
    wandb.finish()


if __name__ == "__main__":

    output_dir = "results/first_try"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env = setup_environment()
    state_size = len(env.state_space.shape)
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)

    if not load_latest_agent(agent, output_dir):
        agent = Agent(state_size, action_size)

    batch_size = 256  # increase by powers of 2
    num_episodes = 10000  # Number of episodes to simulate
    num_iterations = 10  # Number of steps per episode

    train_agent(agent, env, num_episodes, num_iterations, batch_size, output_dir)
