from abc import ABC, abstractmethod
import numpy

from menv.sm_seq import create_gaussian_smooth_seq

create_return_seq = create_gaussian_smooth_seq

class LpComputer(ABC):
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.timestep = 0
        self.returns = [create_return_seq() for _ in range(self.num_envs)]
        self.lps = numpy.zeros((self.num_envs))

    def __call__(self, returns):
        self.timestep += 1
        for env_id, returnn in returns.items():
            self.returns[env_id].append(self.timestep, returnn)
            self._compute_lp(env_id)
        return self.lps

    @abstractmethod
    def _compute_lp(self, env_id):
        pass

class TSLpComputer(LpComputer):
    def __init__(self, num_envs, α):
        super().__init__(num_envs)

        self.α = α

    @abstractmethod
    def _compute_direct_lp(self, env_id):
        pass

    def _compute_lp(self, env_id):
        lp = self._compute_direct_lp(env_id)
        if lp is not None:
            self.lps[env_id] = self.α * lp + (1 - self.α) * self.lps[env_id]

class OnlineLpComputer(TSLpComputer):
    def _compute_direct_lp(self, env_id):
        timesteps, returns = self.returns[env_id][-2:]
        if len(returns) >= 2:
            return numpy.polyfit(timesteps, returns, 1)[0]

class WindowLpComputer(TSLpComputer):
    def __init__(self, num_envs, α, K):
        super().__init__(num_envs, α)

        self.K = K

    def _compute_direct_lp(self, env_id):
        timesteps, returns = self.returns[env_id][-self.K:]
        if len(timesteps) >= 2:
            return numpy.polyfit(timesteps, returns, 1)[0]

class LinregLpComputer(LpComputer):
    def __init__(self, num_envs, K):
        super().__init__(num_envs)

        self.K = K

    def _compute_lp(self, env_id):
        timesteps, returns = self.returns[env_id][-self.K:]
        if len(timesteps) >= 2:
            self.lps[env_id] = numpy.polyfit(timesteps, returns, 1)[0]