import gym
import gym_minigrid
import addition_env


def make_env(env_key, seed):
    env = gym.make(env_key)
    env.seed(seed)
    return env


def make_addition_env(seq_len, number_of_digits, seed):
    env = addition_env.AdditionEnvironment(seq_len, number_of_digits, seed)
    return env
