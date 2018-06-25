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
    """A distribution computer based on learning progress and some
    potential.

    It associates an attention a_i to each environment i that is a
    combinaison of the learning progress and potential of this
    environment, i.e. a_i = lp_i + \alpha pot_i where \alpha is the
    potential coefficient."""

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