# PART 1: Imports
import os
import gymnasium as gym
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.cuda.amp import GradScaler, autocast


# PART 2: DQN Agent Defined
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.002
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model().to(device)
       
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.scaler = GradScaler()  # Initialize the GradScaler


    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.GELU(),
            #nn.ReLU(),
            nn.Linear(24, 24),
            nn.GELU(),
            #nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(device)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
        # return np.argmax(act_values.cpu().detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.LongTensor([action]).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            done = torch.FloatTensor([done]).to(device)


          # Forward pass with autocast
            with autocast():
                q_values = self.model(state)
                next_q_values = self.model(next_state)
                q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
                next_q_value = next_q_values.max(1)[0]
                expected_q_value = reward + self.gamma * next_q_value * (1 - done)
                loss = (q_value - expected_q_value.detach()).pow(2).mean()

            # Scaled backpropagation
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.to(device)

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# PART 3: Create Gym Environment and Assign Vars
env = gym.make('CartPole-v1',render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 1024

output_dir = 'results/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# PART 4: Training Loop
episodes = 1000
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state[0], [1, state[0].shape[0]])
    state = torch.FloatTensor(state)
    for time in range(500):  # set to a high number
        action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        next_state = np.reshape(next_state, [1, next_state.shape[0]])
        reward = reward if not done else -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
    agent.replay(batch_size)
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + "{:04d}".format(e) + ".hdf5")


# PART 5: Testing Agent
agent.load(output_dir + "weights_" + "{:04d}".format(50) + ".hdf5")
for e in range(10):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}/10, Score: {time}")
            break
env.close()