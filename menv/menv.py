from abc import ABC, abstractmethod
from gym.core import Env
import numpy

class MEnv(ABC):
    def __init__(self, G, compute_lp=None, compute_dist=None):
        self.G = G
        self.compute_lp = compute_lp
        self.compute_dist = compute_dist

        self.envs = list(self.G.nodes)
        self.num_envs = len(self.envs)
        self.returnn = None
        self._reset_returns()
        self.dist = numpy.ones((self.num_envs))/self.num_envs
        self.reset()
    
    def __getattr__(self, key):
        return getattr(self.env, key)

    def _select_env(self):
        self.env_id = numpy.random.choice(range(self.num_envs), p=self.dist)
        self.env = self.envs[self.env_id]

    def _reset_returns(self):
        self.returns = {env_id: [] for env_id in range(self.num_envs)}

    def _synthesize_returns(self):
        self.synthesized_returns = {}
        for env_id, returnn in self.returns.items():
            if len(returnn) > 0:
                self.synthesized_returns[env_id] = numpy.mean(returnn)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.returnn += reward
        return obs, reward, done, info

    def update_dist(self):
        self._synthesize_returns()
        self._reset_returns()
        if self.compute_lp is not None and self.compute_dist is not None:
            self.lps = self.compute_lp(self.synthesized_returns)
            self.dist = self.compute_dist(self.lps)

    def reset(self):
        if self.returnn is not None:
            self.returns[self.env_id].append(self.returnn)
        self.returnn = 0
        self._select_env()
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)