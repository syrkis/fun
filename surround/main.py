# main.py
#     rl based go player
# by: Noah Syrkis\

# imports
from src import Agent, Model, game, get_env
import pickle


# main
def main():
    env = get_env()
    env.reset()
    obs = env.observe(env.agents[0])
    agents = {agent: Agent(Model(env.observation_space(agent), env.action_space(agent))) for agent in env.agents}
    game(env, agents, episodes=10, batch_size=32)
    pickle.dump(agents, open("agent.pkl", "wb"))


# run
if __name__ == '__main__':
    main()
