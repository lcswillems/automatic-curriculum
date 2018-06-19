from abc import ABC, abstractmethod
import numpy
import networkx as nx

class DistComputer(ABC):
    @abstractmethod
    def __call__(self, returns):
        pass

class LpDistComputer(DistComputer):
    def __init__(self, compute_lp, create_dist):
        self.compute_lp = compute_lp
        self.create_dist = create_dist

    def __call__(self, returns):
        self.lps = self.compute_lp(returns)
        self.attentions = numpy.absolute(self.lps)
        dist = self.create_dist(self.attentions)

        return dist

class LpRwpotDistComputer(DistComputer):
    def __init__(self, compute_lp, compute_rwpot, create_dist, pot_coeff):
        self.compute_lp = compute_lp
        self.compute_rwpot = compute_rwpot
        self.create_dist = create_dist
        self.pot_coeff = pot_coeff

    def __call__(self, returns):
        self.lps = self.compute_lp(returns)
        self.rwpots = self.compute_rwpot(returns)
        self.attentions = numpy.absolute(self.lps) + self.pot_coeff * self.rwpots
        dist = self.create_dist(self.attentions)

        return dist

class LpLppotDistComputer(DistComputer):
    def __init__(self, G, compute_lp, compute_rwpot, create_dist, pot_coeff):
        self.G = G
        self.compute_lp = compute_lp
        self.compute_rwpot = compute_rwpot
        self.create_dist = create_dist
        self.pot_coeff = pot_coeff

    def __call__(self, returns):
        self.lps = self.compute_lp(returns)
        self.rwpots = self.compute_rwpot(returns)
        self.lppots = numpy.zeros(len(self.G.nodes))
        for env_id in range(len(self.lppots)):
            predecessors = list(self.G.predecessors(env_id))
            predecessors_rwpot = numpy.mean(self.rwpots[predecessors]) if len(predecessors) > 0 else 0
            self.lppots[env_id] = self.rwpots[env_id] - predecessors_rwpot
        self.attentions = numpy.absolute(self.lps) + self.pot_coeff * numpy.absolute(self.lppots)
        dist = self.create_dist(self.attentions)

        return dist