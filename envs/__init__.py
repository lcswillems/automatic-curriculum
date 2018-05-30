import re

import envs.singleenv
import envs.multienv

def get_envs(s, seed, num_procs=1):
    loader = "get_several_" + s
    if loader in dir(envs.singleenv):
        return getattr(envs.singleenv, loader)(seed, num_procs)
    elif loader in dir(envs.multienv):
        return getattr(envs.multienv, loader)(seed, num_procs)
    else:
        raise ValueError("{} is not a correct environment.".format(s))

def get_senv_ids():
    senv_ids = []
    for func in dir(envs.singleenv):
        m = re.search("get_several_(.+)", func)
        if m:
            senv_ids.append(m.group(1))
    return senv_ids