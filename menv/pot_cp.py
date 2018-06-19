from abc import ABC, abstractmethod
import numpy

class PotComputer(ABC):
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.returns = [[] for _ in range(self.num_envs)]
        self.pots = None

    def __call__(self, returns):
        for env_id, returnn in returns.items():
            self.returns[env_id].append(returnn)
            self._compute_pot(env_id)
        return self.pots

    @abstractmethod
    def _compute_pot(self, env_id):
        pass

class RwpotPotComputer(PotComputer):
    def __init__(self, num_envs, K, returns=None, max_returns=None):
        super().__init__(num_envs)

        self.K = K
        returns = [float("+inf")]*self.num_envs if returns is None else returns
        self.max_returns = [float("-inf")]*self.num_envs if max_returns is None else max_returns

        self.rwpots = numpy.zeros((self.num_envs))
        self.pots = self.rwpots

        for env_id in range(len(self.rwpots)):
            returnn = returns[env_id]
            max_return = self.max_returns[env_id]
            self.rwpots[env_id] = max(max_return - returnn, 0)

    def _compute_pot(self, env_id):
        returns = self.returns[env_id][-self.K:]
        returnn = numpy.mean(returns)
        max_return = max(self.max_returns[env_id], returnn)
        self.rwpots[env_id] = max_return - returnn
        if len(returns) >= self.K:
            self.max_returns[env_id] = max_return

class LppotPotComputer(RwpotPotComputer):
    def __init__(self, G, K, returns=None, max_returns=None):
        super().__init__(len(G.nodes), K, returns, max_returns)

        self.G = G

        self.lppots = numpy.zeros((self.num_envs))
        self.pots = self.lppots

    def _compute_pot(self, env_id):
        super()._compute_pot(env_id)

        predecessors = list(self.G.predecessors(env_id))
        predecessors_rwpot = numpy.mean(self.rwpots[predecessors]) if len(predecessors) > 0 else 0
        self.lppots[env_id] = self.rwpots[env_id] - predecessors_rwpot