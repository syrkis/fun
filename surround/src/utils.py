# utils.py
#     utils
# by: Noah Syrkis

# imports
from pettingzoo.atari import surround_v2
import supersuit


# return env
def get_env():
    env = surround_v2.env(render_mode="human")
    env = supersuit.max_observation_v0(env, 2)
    # env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    # env = supersuit.frame_stack_v1(env, 4)
    return env

#%%
