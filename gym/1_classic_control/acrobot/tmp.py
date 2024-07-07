{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3fe4de1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import gym\n",
    "import torch\n",
    "from collections import deque\n",
    "import random\n",
    "\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b03b18a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the neural network model\n",
    "class DQN(nn.Module):\n",
    "    def __init__(self, input_size, output_size):\n",
    "        super(DQN, self).__init__()\n",
    "        self.fc1 = nn.Linear(input_size, 64)\n",
    "        self.fc2 = nn.Linear(64, 64)\n",
    "        self.fc3 = nn.Linear(64, output_size)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = F.relu(self.fc2(x))\n",
    "        x = self.fc3(x)\n",
    "        return x\n",
    "\n",
    "# Define the DQN agent\n",
    "class DQNAgent:\n",
    "    def __init__(self, state_size, action_size):\n",
    "        self.state_size = state_size\n",
    "        self.action_size = action_size\n",
    "        self.memory = deque(maxlen=10000)\n",
    "        self.batch_size = 64\n",
    "        self.gamma = 0.99\n",
    "        self.epsilon = 1.0\n",
    "        self.epsilon_decay = 0.995\n",
    "        self.epsilon_min = 0.01\n",
    "        self.model = DQN(state_size, action_size)\n",
    "        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)\n",
    "\n",
    "    def remember(self, state, action, reward, next_state, done):\n",
    "        self.memory.append((state, action, reward, next_state, done))\n",
    "\n",
    "    def act(self, state):\n",
    "        if random.random() <= self.epsilon:\n",
    "            return random.randrange(self.action_size)\n",
    "        else:\n",
    "            with torch.no_grad():\n",
    "                state = torch.FloatTensor(state).unsqueeze(0)\n",
    "                q_values = self.model(state)\n",
    "                return torch.argmax(q_values).item()\n",
    "\n",
    "    def replay(self):\n",
    "        if len(self.memory) < self.batch_size:\n",
    "            return\n",
    "        batch = random.sample(self.memory, self.batch_size)\n",
    "        states, actions, rewards, next_states, dones = zip(*batch)\n",
    "        states = torch.FloatTensor(states)\n",
    "        actions = torch.LongTensor(actions)\n",
    "        rewards = torch.FloatTensor(rewards)\n",
    "        next_states = torch.FloatTensor(next_states)\n",
    "        dones = torch.BoolTensor(dones)\n",
    "\n",
    "        q_values = self.model(states)\n",
    "        next_q_values = self.model(next_states)\n",
    "        q_targets = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (~dones)\n",
    "        q_targets = q_targets.unsqueeze(1)\n",
    "\n",
    "        self.optimizer.zero_grad()\n",
    "        loss = F.mse_loss(q_values.gather(1, actions.unsqueeze(1)), q_targets)\n",
    "        loss.backward()\n",
    "        self.optimizer.step()\n",
    "\n",
    "        if self.epsilon > self.epsilon_min:\n",
    "            self.epsilon *= self.epsilon_decay\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a49981cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create the environment\n",
    "env = gym.make('Acrobot-v1')\n",
    "state_size = env.observation_space.shape[0]\n",
    "action_size = env.action_space.n\n",
    "\n",
    "# Create the DQN agent\n",
    "agent = DQNAgent(state_size, action_size)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f83f4b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the agent\n",
    "num_episodes = 1000\n",
    "for episode in range(num_episodes):\n",
    "    state = env.reset()\n",
    "    done = False\n",
    "    total_reward = 0\n",
    "\n",
    "    while not done:\n",
    "        action = agent.act(state)\n",
    "        next_state, reward, done, _ = env.step(action)\n",
    "        agent.remember(state, action, reward, next_state, done)\n",
    "        state = next_state\n",
    "        total_reward += reward\n",
    "\n",
    "    agent.replay()\n",
    "\n",
    "    print(f\"Episode: {episode+1}, Total Reward: {total_reward}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db7f177b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test the agent\n",
    "state = env.reset()\n",
    "done = False\n",
    "total_reward = 0\n",
    "\n",
    "while not done:\n",
    "    action = agent.act(state)\n",
    "    next_state, reward, done, _ = env.step(action)\n",
    "    state = next_state\n",
    "    total_reward += reward\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f73b990f-0bbd-4d8c-b555-b245dfb469a6",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'gym'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[2], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgym\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mcollections\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m deque\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'gym'"
     ]
    }
   ],
   "source": [
    "print(f\"Test Total Reward: {total_reward}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
