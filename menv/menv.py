from abc import ABC, abstractmethod
from gym.core import Env
import numpy

class MEnv(ABC):
    def __init__(self, G, compute_lr=None, compute_distrib=None):
        self.G = G
        self.compute_lr = compute_lr
        self.compute_distrib = compute_distrib

        self.menv_logger = None
        self.envs = list(self.G.nodes)
        self.num_envs = len(self.envs)
        self.env = None
        self.env_id = None
        self.returnn = None
        self.lrs = numpy.zeros((self.num_envs))
        self.distrib = numpy.ones((self.num_envs))/self.num_envs
        self.reset()
    
    def __getattr__(self, key):
        return getattr(self.env, key)

    def _select_env(self):
        self.env_id = numpy.random.choice(range(self.num_envs), p=self.distrib)
        self.env = self.envs[self.env_id]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.returnn += reward
        return obs, reward, done, info

    def reset(self):
        if self.returnn is not None:
            if self.compute_distrib is not None:
                self.lrs[self.env_id] = self.compute_lr(self.env_id, self.returnn)
            if self.compute_distrib is not None:
                self.distrib = self.compute_distrib(self.lrs)

            if self.menv_logger is not None:
                self.menv_logger.log()

        self.returnn = 0
        self._select_env()
        
        return self.env.reset()
    
    def render(self, mode="human"):
        return self.env.render(mode)