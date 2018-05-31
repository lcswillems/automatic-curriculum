import envs
from envs.multienv import MEnv_OnlineGreedy
from envs.singleenv import SEnv

def get_several(getter, seed, num_procs):
    return [getter(seed + shift) for shift in range(num_procs)]

def get_several_SEnv_D1LnInBn(seed, num_procs):
    return get_several(get_SEnv_D1LnInBn, seed, num_procs)

def get_SEnv_D1LnInBn(seed):
    return SEnv(1, 0, 0, 0, seed)

def get_several_SEnv_D1LaInBn(seed, num_procs):
    return get_several(get_SEnv_D1LaInBn, seed, num_procs)

def get_SEnv_D1LaInBn(seed):
    return SEnv(1, 1, 0, 0, seed)
        
def get_several_SEnv_D1LnInBa(seed, num_procs):
    return get_several(get_SEnv_D1LnInBa, seed, num_procs)

def get_SEnv_D1LnInBa(seed):
    return SEnv(1, 0, 0, 1, seed)
        
def get_several_SEnv_D1LaInBa(seed, num_procs):
    return get_several(get_SEnv_D1LaInBa, seed, num_procs)

def get_SEnv_D1LaInBa(seed):
    return SEnv(1, 1, 0, 1, seed)
        
def get_several_SEnv_D1LaIaBn(seed, num_procs):
    return get_several(get_SEnv_D1LaIaBn, seed, num_procs)

def get_SEnv_D1LaIaBn(seed):
    return SEnv(1, 1, 1, 0, seed)
        
def get_several_SEnv_D2LnInBn(seed, num_procs):
    return get_several(get_SEnv_D2LnInBn, seed, num_procs)

def get_SEnv_D2LnInBn(seed):
    return SEnv(2, 0, 0, 0, seed)
        
def get_several_SEnv_D4LnInBn(seed, num_procs):
    return get_several(get_SEnv_D4LnInBn, seed, num_procs)

def get_SEnv_D4LnInBn(seed):
    return SEnv(4, 0, 0, 0, seed)

def get_several_SEnv_D4LuIuBu(seed, num_procs):
    return get_several(get_SEnv_D4LuIuBu, seed, num_procs)

def get_SEnv_D4LuIuBu(seed):
    return SEnv(4, 0.5, 0.5, 0.5, seed)

def get_several_MEnv_OnlineGreedy(seed, num_procs):
    menvs = []
    for shift in range(num_procs):
        senvs = envs.get_senvs(seed + shift)
        menvs.append(MEnv_OnlineGreedy(senvs, 0.1, 0.1))
    return menvs