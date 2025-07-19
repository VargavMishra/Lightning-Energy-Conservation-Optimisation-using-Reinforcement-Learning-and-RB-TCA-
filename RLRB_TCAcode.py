import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#lesser the reward more is the energy energy dissipation


class LightningEnv:
    def __init__(self):
        self.state = np.random.uniform(100, 1000)
        self.max_energy = 10000

    def step(self, action):
        """Actions: 0 = Store, 1 = Convert, 2 = Dissipate"""
        reward, done = 0, False
        lightning_energy = np.random.uniform(500, 3000)
        if action == 0:  # Store
            self.state += min(lightning_energy, self.max_energy - self.state)
            reward = self.state / self.max_energy
        elif action == 1:
            converted = 0.8 * lightning_energy
            self.state -= converted
            reward = converted / lightning_energy
        elif action == 2:
            self.state -= min(lightning_energy, self.state)
            reward = -0.2
        self.state = max(0, min(self.state, self.max_energy))
        if self.state == self.max_energy or self.state == 0:
            done = True
        return self.state, reward, done

# Q-Network for Reinforcement Learning
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#(RB-TCA)
def rule_based_control(state):
    if state < 3000:
        return 0
    elif 3000 <= state < 8000:
        return 1
    else:
        return 2

#Reinforcement Learning Agent
class RLAgent:
    def __init__(self, state_size, action_size):
        self.qnetwork = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=0.001)
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0
        self.gamma = 0.95

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])  # Random action
        state_tensor = torch.tensor([state], dtype=torch.float32)
        action_values = self.qnetwork(state_tensor)
        return torch.argmax(action_values).item()

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in batch:
            target = reward + self.gamma * torch.max(self.qnetwork(torch.tensor([next_state], dtype=torch.float32)))
            output = self.qnetwork(torch.tensor([state], dtype=torch.float32)).squeeze(0)[action]
            loss = (target - output) ** 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training Loop
env = LightningEnv()
agent = RLAgent(state_size=1, action_size=3)

for episode in range(500):
    state = env.state
    done = False
    episode_reward = 0
    energy_sum = 0
    action_counts = [0, 0, 0]

    while not done:
        if random.random() < 0.3:
            action = rule_based_control(state)
        else:
            action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.memory.append((state, action, reward, next_state))
        agent.train()
        state = next_state
        episode_reward += reward
        energy_sum += state
        action_counts[action] += 1

    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    if episode % 50 == 0:
        if energy_sum > 0:
            avg_reward = episode_reward / (energy_sum/energy_sum)
            avg_energy = energy_sum / (energy_sum/energy_sum)
        else:
            avg_reward = 0
            avg_energy = 0

        print(
            f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
            f"Avg Energy: {avg_energy:.2f}, Epsilon: {agent.epsilon:.3f}, "
            f"Actions: Store={action_counts[0]}, Convert={action_counts[1]}, Dissipate={action_counts[2]}"
        )
print("******************************************************************************************")
print("Training Complete: RL + Rule-Based Threshold Control Optimization Done!")
