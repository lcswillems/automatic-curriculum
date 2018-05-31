import envs
import envs.menvs.menvs as menvs

def get_several_MEnv_OnlineGreedy(seed, num_procs):
    return [
        menvs.MEnv_OnlineGreedy(envs.get_senvs(seed + shift), 0.1, 0.1, seed)
        for shift in range(num_procs)
    ]