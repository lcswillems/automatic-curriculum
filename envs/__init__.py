import re

import envs.senvs.generate
import envs.menvs.generate
from envs.menvs.tb_logger import TbLogger

def get_envs(s, seed, num_procs):
    loader = "get_several_" + s
    if loader in dir(envs.senvs.generate):
        return getattr(envs.senvs.generate, loader)(seed, num_procs)
    elif loader in dir(envs.menvs.generate):
        return getattr(envs.menvs.generate, loader)(seed, num_procs)
    else:
        raise ValueError("{} is not a correct environment.".format(s))

def get_env(s, seed):
    return get_envs(s, seed, 1)[0]

def get_senv_names():
    senv_names = []
    for func in sorted(dir(envs.senvs.generate)):
        m = re.search("get_several_(.+)", func)
        if m:
            senv_names.append(m.group(1))
    return senv_names

def get_senvs(seed):
    senv_names = get_senv_names()
    return [get_env(senv_name, seed) for senv_name in senv_names]