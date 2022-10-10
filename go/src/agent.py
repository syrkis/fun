# agent.py
#     agent class
# by: Noah Syrkis

# imports
import torch
import numpy as np

from collections import deque
import random


# agent class
class Agent:
    def __init__(self, model, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, gamma=0.95, memory_size=2000):
        self.model = model
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.model.output_space)
        return torch.argmax(self.model.predict(state)).item()

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        experience = (torch.from_numpy(state).float(), action, reward, torch.from_numpy(next_state).float(), done)
        self.memory.append(experience)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        loss = 0
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model.predict(next_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            loss += self.model.fit(state, target_f)
        print("Loss:", loss)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("Epsilon", self.epsilon)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))