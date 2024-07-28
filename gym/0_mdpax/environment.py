#!/home/titan/workspace/mlos/venvs/jax/bin/python3

import jax
import jax.numpy as jnp
from jax import jit, random

from typing import Any, Callable

from gymnasium import spaces, Env


class EnvironmentConfig:
    def __init__(
        self,
        state_space: Any,
        action_space: Any,
        initial_state: Any,
        target_state: Any,
        reward_function: Callable,
        transition_function: Callable,
    ):
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

        # setups, overide the functions with jax.jit
        self._transition = jax.jit(
            lambda state, action: self.transition_function(
                state, action, self.state_space.shape
            )
        )
        self._compute_reward = jax.jit(
            lambda state: self.reward_function(state, self.target_state)
        )
        self._is_done = jax.jit(lambda state: jnp.array_equal(state, self.target_state))

        # Modify gym spaces
        # self.observation_space = spaces.Box(low=-jnp.inf, high=jnp.inf, shape=self.state_space.shape, dtype=jnp.float32)

    def reset(self):
        """Resets the environment to the initial state and returns the initial observation."""
        self.current_state = self.config.initial_state
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
        next_state = self._transition(self.current_state, action)
        reward = self._compute_reward(next_state)
        done = self._is_done(next_state)
        self.current_state = next_state
        return next_state, reward, done, {}


# EXTRA METHODS INSTEAD OF ONE LINE FUNCTIONS
# --------------------------------------------
"""
@jax.jit
def f_transition(self, state, action):
    return self.transition_function(state, action, self.state_space.shape)

@jax.jit
def f_compute_reward(self, state):
    return self.reward_function(state, self.target_state)
@jax.jit
def f_is_done(self, state):
    return  jnp.array_equal(self.state, self.target_state)
"""
# --------------------------------------------


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
