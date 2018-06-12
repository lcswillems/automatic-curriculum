import gym
import babyai

import scenvs

def make_env(env_key, seed):
    env = gym.make(env_key)
    env.seed(seed)
    return env

def make_envs(env_key, seed, num_procs):
    envs = []
    for i in range(num_procs):
        envs.append(make_env(env_key, seed + i))
    return envs