# main.py
#     rl based magent player
# by: Noah Syrkis\

# imports
from src import Agent, Model, game
from pettingzoo.magent import battle_v4


# main
def main():
    env = battle_v4.env()
    battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
                  dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
                  max_cycles=1000, extra_features=False)
    env.reset()
    print(env.agents)
    exit()
    agents = {agent: Agent(Model(env.observation_space(agent), env.action_space(agent))) for agent in env.agents}
    game(env, agents, episodes=100, batch_size=32)


# run
if __name__ == '__main__':
    main()
