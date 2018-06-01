import gym
import babyai

import scenvs

def make_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)
    return env

def make_envs(env_id, seed, num_procs):
    envs = []
    for i in range(num_procs):
        envs.append(make_env(env_id, seed + i))
    return envs