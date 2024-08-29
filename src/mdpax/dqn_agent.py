#!/home/titan/workspace/mlos/venvs/jax/bin/python3

import os
import time
import logging
import random as rd


import numpy as np
from collections import deque


import jax
from jax import jit, random, vmap, numpy as jnp

import flax
from flax import serialization, linen as nn
from flax.training import train_state


import optax


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
        self.key, self.subkey = self.hydra(self.key)

        self.model = DQNModel(state_size, action_size)
        self.params = self.model.init(self.subkey, jnp.ones((1, state_size)))
        self.optimizer = optax.adam(self.learning_rate)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=self.params, tx=self.optimizer)

        # cant believe this worked
        self.model.apply = jit(self.model.apply)

    def hydra(self, key):
        new_key, subkey = random.split(key)
        return new_key, subkey

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return rd.randrange(self.action_size)
        state = jnp.array(state)
        q_values = self.model.apply(self.state.params, state)
        return jnp.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # sample and assign
        minibatch = rd.sample(self.memory, batch_size)
        states = jnp.array([experience[0] for experience in minibatch], dtype=jnp.float32)
        actions = jnp.array([experience[1] for experience in minibatch], dtype=jnp.int32)
        rewards = jnp.array([experience[2] for experience in minibatch], dtype=jnp.int32)
        next_states = jnp.array([experience[3] for experience in minibatch], dtype=jnp.float32)
        dones = jnp.array([experience[4] for experience in minibatch], dtype=jnp.bool_)

        # compute all the q values for all the actions
        @jax.jit
        def compute_target_q_values(rewards, gamma, futures, dones):
            return rewards + gamma * futures * (1 - dones)

        # compute what model thinks the future q values are
        futures = jnp.max(self.model.apply(self.state.params, next_states), axis=-1)
        # compute the target q values
        target_q_values = compute_target_q_values(rewards, self.gamma, futures, dones)

        # define the loss function
        def loss_fn(params, state, action, target):
            q_values = self.model.apply(params, state)
            q_value = q_values[action]
            loss = jnp.mean((target - q_value) ** 2)
            return loss

        # compute the gradient function
        grad_fn = jax.grad(loss_fn)
        # vmap the gradient function (vectorize the gradient function)
        vmap_grad_fn = vmap(grad_fn, in_axes=(None, 0, 0, 0))

        # for serial => grads = jax.grad(loss_fn)(states, actions, target_q_values)
        # compute the gradients using grad_fn with vmap_grad_fn
        grads = vmap_grad_fn(self.state.params, states, actions, target_q_values)
        # average the gradients
        average_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
        # apply the gradients
        self.state = self.state.apply_gradients(grads=average_grads)

        loss = jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))(self.state.params, states, actions, target_q_values))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def load(self, name):
        with open(name, "rb") as f:
            bytes_input = f.read()
            self.state = serialization.from_bytes(train_state.TrainState, bytes_input)

    def save(self, name):
        with open(name, "wb") as f:
            bytes_output = serialization.to_bytes(self.state)
            f.write(bytes_output)

    """
    def load(self, name):
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=jnp.load(name), tx=self.optimizer)

    def save(self, name):
        jnp.save(name, self.state.params)
    """
