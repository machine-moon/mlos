#!/home/titan/workspace/mlos/venvs/jax/bin/python3

from environment import EnvironmentConfig, create_environment
import jax
import jax.numpy as jnp
from jax import jit, random
from gymnasium import spaces


def example_reward_function(state, goal_state):
    """Define your reward logic here."""
    # if jnp.array_equal(state, jnp.array([4, 4])):
    #    return 10
    # else:
    #    return -1
    return jnp.where(jnp.array_equal(state, goal_state), 10, -1)


def example_transition_function(state, action, state_space_shape):
    """Define your state transition logic here."""
    x, y = state

    def move_up(_):
        return jax.lax.max(0, x - 1), y

    def move_down(_):
        return jax.lax.min(state_space_shape[0] - 1, x + 1), y

    def move_left(_):
        return x, jax.lax.max(0, y - 1)

    def move_right(_):
        return x, jax.lax.min(state_space_shape[1] - 1, y + 1)

    x, y = jax.lax.switch(action, [move_up, move_down, move_left, move_right], None)
    return jnp.array([x, y])


# Define the environment configuration

env_width, env_height = 10, 10
action_space_n = 4

dtype = jnp.int32

state_space = spaces.Box(
    low=0, high=max(env_width, env_height), shape=(env_width, env_height), dtype=dtype
)

action_space = spaces.Discrete(action_space_n)

initial_state = jnp.array([0, 0], dtype=dtype)
target_state = jnp.array([4, 4], dtype=dtype)


# Create the environment by creating an EnvironmentConfig object and passing it to the create_environment function

config = EnvironmentConfig(
    state_space=state_space,
    action_space=action_space,
    initial_state=initial_state,
    target_state=target_state,
    reward_function=jit(example_reward_function),
    transition_function=jit(example_transition_function),
)


# creatimport os
import os
import numpy as np
from collections import deque
import flax
from flax import linen as nn

env = create_environment(config)
state_size = len(state_space.shape)
action_size = env.action_space.n


batch_size = 32  # increase by powers of 2
key = random.PRNGKey(0)
num_episodes = 100  # Number of episodes to simulate
num_iterations = 200  # Number of steps per episode


output_dir = "results/mdp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for e in range(num_episodes):

    state, reward, done = env.reset()
    print(f"episode: {e} state: {state}, reward: {reward}, action: none done: {done}")

    episode_reward = 0  # Track total reward per episode

    for _ in range(num_iterations):
        key, subkey = random.split(key)
        action = random.choice(
            subkey, env.action_space.n
        )  # alternatively, you can use env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward  # Accumulate reward per step
        state = next_state
        print(
            f"episode: {e} state: {state}, reward: {reward}, action: {action}, done: {done}"
        )
        if done or _ == 199:
            print(f"Total Reward: {episode_reward}")  # Print total reward per episode
            episode_reward = 0  # Reset episode reward
            break
