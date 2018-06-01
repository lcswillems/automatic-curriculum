from abc import ABC, abstractmethod
from gym.core import Env
import numpy

from envs.menvs.tb_logger import TbLogger

class MEnv(ABC):
    def __init__(self, envs_nxgraph, seed=None):
        self.envs_nxgraph = envs_nxgraph
        self.rng = numpy.random.RandomState(seed)

        self.tb_logger = None
        self.envs = list(self.envs_nxgraph.nodes)
        self.num_envs = len(self.envs)
        self.returns = [[] for _ in range(self.num_envs)]
        self.lrs = numpy.zeros(self.num_envs)
        self.distrib = numpy.ones(self.num_envs)/self.num_envs
        self.returnn = None
        self.env_id = None
        self.env = None
        self.reset()
    
    def __getattr__(self, key):
        return getattr(self.env, key)

    def _select_env(self):
        self.env_id = self.rng.choice(range(self.num_envs), p=self.distrib)
        self.env = self.envs[self.env_id]
    
    @abstractmethod
    def _update_lrs(self):
        pass
    
    @abstractmethod
    def _update_distrib(self):
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.returnn += reward
        return obs, reward, done, info

    def reset(self):
        if self.returnn is not None:
            self._update_lrs()
            self._update_distrib()

            if self.tb_logger is not None:
                self.tb_logger.log()

        self.returnn = 0
        self._select_env()
        
        return self.env.reset()
    
    def render(self, mode="human"):
        return self.env.render(mode)

class MEnv_OnlineGreedy(MEnv):
    def __init__(self, envs_nxgraph, ε, α, seed=None):
        self.ε = ε
        self.α = α

        super().__init__(envs_nxgraph, seed)
    
    def _update_lrs(self):
        self.returns[self.env_id].append(self.returnn)
        returns = self.returns[self.env_id]
        if len(returns) >= 2:
            lr = returns[-1] - returns[-2]
            self.lrs[self.env_id] = self.α * lr + (1 - self.α) * self.lrs[self.env_id]
    
    def _update_distrib(self):
        abs_lrs = numpy.absolute(self.lrs)
        env_id = self.rng.choice(numpy.flatnonzero(abs_lrs == abs_lrs.max()))

        self.distrib = self.ε*numpy.ones((self.num_envs))/self.num_envs
        self.distrib[env_id] += 1-self.ε

class MEnv_Uniform(MEnv):
    def __init__(self, envs_nxgraph, seed=None):
        super().__init__(envs_nxgraph, seed)