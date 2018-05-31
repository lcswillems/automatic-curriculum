from abc import ABC, abstractmethod
from collections import deque
from gym.core import Env
import numpy

import envs

class MultiEnv(Env, ABC):
    def __init__(self, envs, capacity=None):
        self.envs = envs
        self.capacity = capacity

        self.num_envs = len(self.envs)
        self.returns = [deque(maxlen=capacity) for _ in range(self.num_envs)]
        self.lrs = [0]*self.num_envs
        self.returnn = None
        self.env_id = None
        self.env = None
        self.reset()
    
    def __getattr__(self, key):
        return getattr(self.env, key)

    @abstractmethod
    def _select_env(self):
        pass
    
    @abstractmethod
    def _update_lr(self):
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.returnn += reward
        return obs, reward, done, info

    def reset(self):
        if self.returnn is not None:
            self._update_lr()
        self.returnn = 0

        self._select_env()
        return self.env.reset()
    
    def render(self, mode="human"):
        return self.env.render(mode)

class MEnv_OnlineGreedy(MultiEnv):
    def __init__(self, envs, ε, α):
        self.ε = ε
        self.α = α

        super().__init__(envs, capacity=2)
    
    def _select_env(self):
        if numpy.random.rand() <= self.ε:
            env_id = numpy.random.randint(0, self.num_envs)
        else:
            abs_lrs = numpy.absolute(self.lrs)
            env_id = numpy.random.choice(numpy.flatnonzero(abs_lrs == abs_lrs.max()))
        self.env_id = env_id
        self.env = self.envs[self.env_id]
    
    def _update_lr(self):
        self.returns[self.env_id].append(self.returnn)
        returns = list(self.returns[self.env_id])
        if len(returns) >= 2:
            lr = returns[-1] - returns[-2]
            self.lrs[self.env_id] = self.α * lr + (1 - self.α) * self.lrs[self.env_id]

def get_several_MEnv_OnlineGreedy(seed, num_procs):
    menvs = []
    for shift in range(num_procs):
        senvs = envs.get_senvs(seed + shift)
        menvs.append(MEnv_OnlineGreedy(senvs, 0.1, 0.1))
    return menvs