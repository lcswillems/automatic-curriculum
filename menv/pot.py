from abc import ABC, abstractmethod
import numpy

class PotComputer(ABC):
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.returns = [[] for _ in range(self.num_envs)]
        self.pots = numpy.zeros((self.num_envs))
    
    def __call__(self, returns):
        for env_id, returnn in returns.items():
            self.returns[env_id].append(returnn)
            self._compute_pot(env_id)
        return self.pots
    
    @abstractmethod
    def _compute_pot(self, env_id):
        pass

class VariablePotComputer(PotComputer):
    def __init__(self, max_returns, K):
        super().__init__(len(max_returns))

        self.max_returns = max_returns
        self.K = K
    
        self.min_returns = [float("+inf")] * self.num_envs

    def _compute_pot(self, env_id):
        returns = self.returns[env_id][-self.K:]
        mean_return = numpy.mean(returns)
        min_return = min(self.min_returns[env_id], mean_return)
        max_return = max(self.max_returns[env_id], mean_return)
        self.pots[env_id] = (mean_return - min_return) / (max_return - min_return)
        if len(returns) >= self.K:
            self.min_returns[env_id] = min_return
            self.max_returns[env_id] = max_return