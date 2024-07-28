#!/home/titan/workspace/mlos/venvs/jax/bin/python3

import os
import numpy as np
from collections import deque
import flax
from flax import linen as nn
from flax.training import train_state

import jax
from jax import jit, random, vmap, numpy as jnp

import optax

import random as rd


class DQNModel(nn.Module):
    state_size: int
    action_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.state_size)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.action_size)(x)
        return q_values


class Agent:
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.key = random.PRNGKey(0)
        self.subkey = self.rng()

        self.model = DQNModel(state_size, action_size)
        self.params = self.model.init(self.subkey, jnp.ones((1, state_size)))
        self.optimizer = optax.adam(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.optimizer
        )

        # cant believe this worked
        self.model.apply = jit(self.model.apply)

    def rng(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return rd.randrange(self.action_size)
        state = jnp.array(state)
        q_values = self.model.apply(self.state.params, state)
        return jnp.argmax(q_values).item()

    def replay(self, batch_size):

        minibatch = rd.sample(self.memory, batch_size)

        states = jnp.array(
            [experience[0] for experience in minibatch], dtype=jnp.float32
        )
        actions = jnp.array(
            [experience[1] for experience in minibatch], dtype=jnp.int32
        )
        rewards = jnp.array(
            [experience[2] for experience in minibatch], dtype=jnp.int32
        )
        next_states = jnp.array(
            [experience[3] for experience in minibatch], dtype=jnp.float32
        )
        dones = jnp.array([experience[4] for experience in minibatch], dtype=jnp.bool_)

        @jax.jit
        def compute_target_q_values(rewards, gamma, futures, dones):
            return rewards + gamma * futures * (1 - dones)

        futures = jnp.max(self.model.apply(self.state.params, next_states), axis=-1)

        gamma = self.gamma
        target_q_values = compute_target_q_values(rewards, gamma, futures, dones)

        def loss_fn(params, state, action, target):
            q_values = self.model.apply(params, state)
            q_value = q_values[action]
            loss = jnp.mean((target - q_value) ** 2)
            return loss

        grad_fn = jax.grad(loss_fn)
        vmap_grad_fn = vmap(grad_fn, in_axes=(None, 0, 0, 0))

        # grads = jax.grad(loss_fn)(states, actions, target_q_values)
        grads = vmap_grad_fn(self.state.params, states, actions, target_q_values)
        average_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)

        self.state = self.state.apply_gradients(grads=average_grads)

    def load(self, name):
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=jnp.load(name), tx=self.optimizer
        )

    def save(self, name):
        jnp.save(name, self.state.params)
