# game.py
#     play game
# by: Noah Syrkis

# imports
import torch
import numpy as np

from tqdm import tqdm


# play game
def game(env, agents, episodes=10, batch_size=32, render=False):
    for episode in tqdm(range(episodes)):
        env.reset()
        observation = env.observe(env.agents[0])
        for agent_id in env.agent_iter():
            agent = agents[agent_id]
            next_observation, reward, done, truncated, info = env.last()
            if done:
                break
            action = agent.get_action(observation) if reward == 0 else 0
            env.step(action)
            agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation
        for agent_id in env.agents:
            agent = agents[agent_id]
            agent.replay(batch_size)

    env.close()
