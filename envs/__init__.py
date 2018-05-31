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

def get_senv_names():
    senv_names = []
    for func in sorted(dir(envs.generate)):
        m = re.search("get_several_(SEnv_.+)", func)
        if m:
            senv_names.append(m.group(1))
    return senv_names

def get_senvs(seed):
    senv_names = get_senv_names()
    return [get_env(senv_name, seed) for senv_name in senv_names]