from envs.env import Env
from envs.multienv import OnlineGreedyEnv

env_ids = ["D1LnInBn", "D1LaInBn", "D1LnInBa", "D1LaInBa", "D1LaIaBn",
           "D2LnInBn", "D4LnInBn", "D4LuIuBu"]

def env_id_to_env(env_id, seed):
    char_to_value = {
        "L": "locked_proba",
        "I": "inbox_proba",
        "B": "blocked_proba",
        "D": "num_doors",
        "n": 0,
        "u": 0.5,
        "a": 1,
        "1": 1,
        "2": 2,
        "4": 4
    }

    params = dict(seed=seed)
    for i in range(0, len(env_id), 2):
        params[char_to_value[env_id[i]]] = char_to_value[env_id[i+1]]
    return Env(**params)

def str_to_envs(s, seed, num_procs=1):
    mode, env_id = s.split("-")

    if mode == "Env":
        assert len(env_id) % 2 == 0, "Id must have an even length."
        return [env_id_to_env(env_id, seed + i) for i in range(num_procs)]
    elif mode == "MultiEnv":
        raise NotImplementedError
    else:
        raise ValueError("Invalid mode: {}".format(mode))