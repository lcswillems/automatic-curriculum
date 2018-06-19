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
    def __init__(self, num_envs, K, min_returns=None, max_returns=None):
        super().__init__(num_envs)

        self.K = K
        self.min_returns = numpy.full(self.num_envs, numpy.inf) if min_returns is None else numpy.array(min_returns)
        self.max_returns = numpy.full(self.num_envs, -numpy.inf) if max_returns is None else numpy.array(max_returns)

        self.rwpots = numpy.zeros((self.num_envs))
        self.pots = self.rwpots

        for env_id in range(len(self.rwpots)):
            RwpotPotComputer._compute_pot(self, env_id)

    def _compute_pot(self, env_id):
        returns = self.returns[env_id][-self.K:]
        min_return = self.min_returns[env_id]
        max_return = self.max_returns[env_id]
        returnn = min_return
        if len(returns) > 0:
            min_return = min(min_return, returnn)
            max_return = max(max_return, returnn)
            returnn = numpy.mean(returns)
        self.rwpots[env_id] = max(max_return - returnn, 0)
        if len(returns) >= self.K:
            self.min_returns[env_id] = min_return
            self.max_returns[env_id] = max_return

class LppotPotComputer(RwpotPotComputer):
    def __init__(self, G, K, min_returns=None, max_returns=None):
        super().__init__(len(G.nodes), K, min_returns, max_returns)

        self.G = G

        self.lppots = numpy.zeros((self.num_envs))
        self.pots = self.lppots

        for env_id in range(len(self.lppots)):
            self._compute_pot(env_id)

    def _compute_pot(self, env_id):
        super()._compute_pot(env_id)

        predecessors = list(self.G.predecessors(env_id))
        if len(predecessors) > 0:
            predecessors_norm = self.max_returns[predecessors] - self.min_returns[predecessors]
            predecessors_pot = numpy.mean(self.rwpots[predecessors] / predecessors_norm)
        else:
            predecessors_pot = 0
        self.lppots[env_id] = self.rwpots[env_id] * (1 - predecessors_pot)