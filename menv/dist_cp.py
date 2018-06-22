from abc import ABC, abstractmethod
import numpy
import networkx as nx

class DistComputer(ABC):
    def __init__(self, return_hists):
        self.return_hists = return_hists

        self.step = 0

    @abstractmethod
    def __call__(self, returns):
        self.step += 1
        for env_id, returnn in returns.items():
            self.return_hists[env_id].append(self.step, returnn)

class LpDistComputer(DistComputer):
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
    def __init__(self, return_hists, compute_lp, compute_pot, create_dist, pot_coef):
        super().__init__(return_hists)

        self.compute_lp = compute_lp
        self.compute_pot = compute_pot
        self.create_dist = create_dist
        self.pot_coef = pot_coef

        self.step = 0

    def __call__(self, returns):
        super().__call__(returns)

        self.lps = self.compute_lp()
        self.pots = self.compute_pot()
        self.attentions = numpy.absolute(self.lps) + self.pot_coef * self.pots
        dist = self.create_dist(self.attentions)

        return dist