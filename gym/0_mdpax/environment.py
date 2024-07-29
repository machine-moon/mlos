#!/home/titan/workspace/mlos/venvs/jax/bin/python3

import jax
import jax.numpy as jnp
from jax import jit, random

from typing import Any, Callable

from gymnasium import spaces, Env


class EnvironmentConfig:
    def __init__(
        self,
        seed: int,
        state_space: Any,
        action_space: Any,
        initial_state: Any,
        target_state: Any,
        reward_function: Callable,
        transition_function: Callable,
    ):
        self.seed = seed
        self.state_space = state_space
        self.action_space = action_space
        self.initial_state = initial_state
        self.target_state = target_state
        self.reward_function = reward_function
        self.transition_function = transition_function


def create_environment(config: EnvironmentConfig):
    return MDPEnvironment(config)


class MDPEnvironment(Env):
    def __init__(self, config: EnvironmentConfig):
        super(MDPEnvironment, self).__init__()
        # Environment configuration
        self.config = config

        self.state_space = config.state_space
        self.action_space = config.action_space

        self.current_state = config.initial_state
        self.target_state = config.target_state

        self.reward_function = config.reward_function
        self.transition_function = config.transition_function
        self.seed = config.seed

        self.key = random.PRNGKey(self.seed)
        self.key, self.subkey = self.hydra(self.key)

        # setups, overide the functions with jax.jit
        self._transition = jax.jit(
            lambda state, action, key: self.transition_function(state, action, self.state_space.shape, key)
        )
        self._compute_reward = jax.jit(lambda state: self.reward_function(state, self.target_state))
        self._is_done = jax.jit(lambda state: (jnp.all(jnp.abs(state - self.target_state) <= 0.1)))

        self._initial = jax.jit(
            lambda key: random.uniform(key, (2,), minval=-1, maxval=1, dtype=self.state_space.dtype)
        )
        # Modify gym spaces
        # self.observation_space = spaces.Box(low=-jnp.inf, high=jnp.inf, shape=self.state_space.shape, dtype=jnp.float32)

    def hydra(self, key):
        # useage: self.key, self.subkey = self.hydra(self.key)
        new_key, subkey = random.split(key)
        return new_key, subkey

    def reset(self):
        """Resets the environment to the initial state and returns the initial observation."""
        self.key, self.subkey = self.hydra(self.key)
        self.current_state = self._initial(self.subkey)
        reward = self._compute_reward(self.current_state)
        done = self._is_done(self.current_state)
        return self.current_state, reward, done

    def step(self, action):
        """Takes a step in the environment based on the action provided.

        Args:
            action: The action to take.

        Returns:
            (next_state, reward, done, info).
        """
        self.key, self.subkey = self.hydra(self.key)
        next_state = self._transition(self.current_state, action, self.subkey)
        reward = self._compute_reward(next_state)
        done = self._is_done(next_state)
        self.current_state = next_state
        return next_state, reward, done, {}


# EXTRA METHODS INSTEAD OF ONE LINE FUNCTIONS
# --------------------------------------------
"""

"""
# --------------------------------------------

"""
# Define state space and action space
env_width, env_height = 5, 5
state_space = spaces.Box(
    low=0,
    high=max(env_width, env_height),
    shape=(env_width, env_height),
    dtype=jnp.int32,
)

action_space_n = 4
action_space = spaces.Discrete(action_space_n)

initial_state = jnp.array([0, 0], dtype=jnp.int32)
target_state = jnp.array([4, 4], dtype=jnp.int32)


# Create the environment
config = EnvironmentConfig(
    state_space=state_space,
    action_space=action_space,
    initial_state=initial_state,
    target_state=target_state,
    reward_function=jit(example_reward_function),
    transition_function=jit(example_transition_function),
)

env = create_environment(config)


# Simulation
key = random.PRNGKey(0)
num_episodes = 10  # Number of episodes to simulate
for e in range(num_episodes):

    state, reward, done = env.reset()
    print(f"episode: {e} state: {state}, reward: {reward}, action: none done: {done}")
    episode_reward = 0  # Track total reward per episode

    for _ in range(200):
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
"""
