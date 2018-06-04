from abc import ABC, abstractmethod
from gym.core import Env
import numpy

class MEnv(ABC):
    def __init__(self, G, compute_lp=None, compute_distrib=None, automatic_update=False):
        self.G = G
        self.compute_lp = compute_lp
        self.compute_distrib = compute_distrib
        self.automatic_update = automatic_update

        self.menv_logger = None
        self.envs = list(self.G.nodes)
        self.num_envs = len(self.envs)
        self.env = None
        self.env_id = None
        self.returnn = None
        self._reset_returns()
        self.lps = None
        self.distrib = numpy.ones((self.num_envs))/self.num_envs
        self.reset()
    
    def __getattr__(self, key):
        return getattr(self.env, key)

    def _select_env(self):
        self.env_id = numpy.random.choice(range(self.num_envs), p=self.distrib)
        self.env = self.envs[self.env_id]

    def _reset_returns(self):
        self.returns = {env_id: [] for env_id in range(self.num_envs)}

    def _synthesize_returns(self):
        for env_id, returnn in self.returns.items():
            if len(returnn) > 0:
                self.returns[env_id] = numpy.mean(returnn)
            else:
                del self.returns[env_id]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.returnn += reward
        return obs, reward, done, info

    def update_distrib(self):
        if self.compute_lp is not None and self.compute_distrib is not None:
            self._synthesize_returns()
            self.lps = self.compute_lp(self.returns)
            self.distrib = self.compute_distrib(self.lps)
            if self.menv_logger is not None:
                self.menv_logger.log()
        self._reset_returns()

    def reset(self):
        if self.returnn is not None:
            self.returns[self.env_id].append(self.returnn)
            if self.automatic_update:
                self.update_distrib()
        self.returnn = 0
        self._select_env()
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)