import envs.senvs.senvs as senvs

def get_several(classs, seed, num_procs):
    return [classs(seed + shift) for shift in range(num_procs)]

def get_several_SEnv_D1LnInBn(seed, num_procs):
    return get_several(senvs.SEnv_D1LnInBn, seed, num_procs)

def get_several_SEnv_D1LaInBn(seed, num_procs):
    return get_several(senvs.SEnv_D1LaInBn, seed, num_procs)
        
def get_several_SEnv_D1LnInBa(seed, num_procs):
    return get_several(senvs.SEnv_D1LnInBa, seed, num_procs)
        
def get_several_SEnv_D1LaInBa(seed, num_procs):
    return get_several(senvs.SEnv_D1LaInBa, seed, num_procs)

def get_several_SEnv_D1LaIaBn(seed, num_procs):
    return get_several(senvs.SEnv_D1LaIaBn, seed, num_procs)

def get_several_SEnv_D2LnInBn(seed, num_procs):
    return get_several(senvs.SEnv_D2LnInBn, seed, num_procs)

def get_several_SEnv_D4LnInBn(seed, num_procs):
    return get_several(senvs.SEnv_D4LnInBn, seed, num_procs)

def get_several_SEnv_D4LuIuBu(seed, num_procs):
    return get_several(senvs.SEnv_D4LuIuBu, seed, num_procs)