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

class LpPotDistComputer(DistComputer):
    def __init__(self, compute_lp, compute_pot, create_dist, pot_coef):
        self.compute_lp = compute_lp
        self.compute_pot = compute_pot
        self.create_dist = create_dist
        self.pot_coef = pot_coef

    def __call__(self, returns):
        self.lps = self.compute_lp(returns)
        self.pots = self.compute_pot(returns)
        self.attentions = numpy.absolute(self.lps) + self.pot_coef * self.pots
        dist = self.create_dist(self.attentions)

        return dist