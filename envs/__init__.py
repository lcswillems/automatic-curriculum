import re

import envs.generate

def get_envs(s, seed, num_procs):
    loader = "get_several_" + s
    if loader in dir(envs.generate):
        return getattr(envs.generate, loader)(seed, num_procs)
    else:
        raise ValueError("{} is not a correct environment.".format(s))

def get_env(s, seed):
    return get_envs(s, seed, 1)[0]

def get_senv_ids():
    senv_ids = []
    for func in sorted(dir(envs.generate)):
        m = re.search("get_several_(SEnv_.+)", func)
        if m:
            senv_ids.append(m.group(1))
    return senv_ids

def get_senvs(seed):
    senv_ids = get_senv_ids()
    return [get_env(senv_id, seed) for senv_id in senv_ids]