# main.py
#     rl based go player
# by: Noah Syrkis\

# imports
from src import Agent, Model, game
from pettingzoo.classic import go_v5


# main
def main():
    env = go_v5.env(board_size=2)
    env.reset()
    obs = env.observe(env.agents[0])
    print(obs['observation'])
    exit()
    agents = {agent: Agent(Model(env.observation_space(agent), env.action_space(agent))) for agent in env.agents}
    game(env, agents, episodes=100, batch_size=32)


# run
if __name__ == '__main__':
    main()
