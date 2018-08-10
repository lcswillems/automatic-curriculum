from abc import ABC, abstractmethod
import numpy
import networkx as nx

class DistComputer(ABC):
    """A distribution computer.

    It receives returns for some environments, updates the return history
    given by each environment and computes a distribution over
    environments given these histories of return."""

    def __init__(self, return_hists):
        self.return_hists = return_hists

        self.step = 0

    @abstractmethod
    def __call__(self, returns):
        self.step += 1
        for env_id, returnn in returns.items():
            self.return_hists[env_id].append(self.step, returnn)

class LpDistComputer(DistComputer):
    """A distribution computer based on learning progress.

    It associates an attention a_i to each environment i that is equal
    to the learning progress of this environment, i.e. a_i = lp_i."""

    def __init__(self, return_hists, compute_lp, create_dist):
        super().__init__(return_hists)

        self.return_hists = return_hists
        self.compute_lp = compute_lp
        self.create_dist = create_dist

    def __call__(self, returns):
        super().__call__(returns)

        self.lps = self.compute_lp()
        self.attentions = numpy.absolute(self.lps)
        dist = self.create_dist(self.attentions)

        return dist

class LpPotDistComputer(DistComputer):
    def __init__(self, return_hists, compute_lp, create_dist, pot_coef,
                 returns, max_returns, K):
        super().__init__(return_hists)

        self.compute_lp = compute_lp
        self.create_dist = create_dist
        self.pot_coef = pot_coef
        self.returns = numpy.array(returns, dtype=numpy.float)
        self.max_returns = numpy.array(max_returns, dtype=numpy.float)
        self.K = K

        self.saved_max_returns = self.max_returns[:]

    def update_returns(self):
        for i in range(len(self.returns)):
            _, returns = self.return_hists[i][-self.K:]
            if len(returns) > 0:
                self.returns[i] = returns[-1]
                mean_return = numpy.mean(returns)
                self.max_returns[i] = max(self.saved_max_returns[i], mean_return)
                if len(returns) >= self.K:
                    self.saved_max_returns[i] = self.max_returns[i]

    def __call__(self, returns):
        super().__call__(returns)

        self.update_returns()

        self.returns = numpy.clip(self.returns, None, self.max_returns)
        self.lps = self.compute_lp()
        self.a_lps = numpy.absolute(self.lps)
        self.pots = self.max_returns - self.returns
        self.attentions = self.a_lps + self.pot_coef * self.pots
        dist = self.create_dist(self.attentions)

        return dist

class LpPotRrDistComputer(DistComputer):
    def __init__(self, return_hists, compute_lp, create_dist, pot_coef,
                 returns, max_returns, K, G):
        super().__init__(return_hists)

        self.compute_lp = compute_lp
        self.create_dist = create_dist
        self.pot_coef = pot_coef
        self.returns = numpy.array(returns, dtype=numpy.float)
        self.max_returns = numpy.array(max_returns, dtype=numpy.float)
        self.K = K
        self.G = G

        self.min_returns = self.returns[:]
        self.saved_min_returns = self.min_returns[:]
        self.saved_max_returns = self.max_returns[:]

    def update_returns(self):
        for i in range(len(self.returns)):
            _, returns = self.return_hists[i][-self.K:]
            if len(returns) > 0:
                self.returns[i] = returns[-1]
                mean_return = numpy.mean(returns)
                self.min_returns[i] = min(self.saved_min_returns[i], mean_return)
                self.max_returns[i] = max(self.saved_max_returns[i], mean_return)
                if len(returns) >= self.K:
                    self.saved_min_returns[i] = self.min_returns[i]
                    self.saved_max_returns[i] = self.max_returns[i]

    def __call__(self, returns):
        super().__call__(returns)

        self.update_returns()

        self.returns = numpy.clip(self.returns, self.min_returns, self.max_returns)
        self.lps = self.compute_lp()
        self.a_lps = numpy.absolute(self.lps)
        self.pots = self.max_returns - self.returns
        self.rrs = (self.returns - self.min_returns) / (self.max_returns - self.min_returns)
        self.filters = numpy.ones(len(self.return_hists))
        for env_id in self.G.nodes:
            predecessors = list(self.G.predecessors(env_id))
            if len(predecessors) > 0:
                self.filters[env_id] = numpy.mean(self.rrs[predecessors])
        self.attentions = (self.a_lps + self.pot_coef * self.pots) * self.filters
        dist = self.create_dist(self.attentions)

        return dist