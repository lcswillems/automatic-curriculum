import gym
import babyai

import scenvs

def make_env(env_key, seed):
    env = gym.make(env_key)
    env.seed(seed)
    return env