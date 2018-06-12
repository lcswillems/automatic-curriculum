from abc import ABC, abstractmethod
import numpy
import networkx as nx

class DistComputer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, lps):
        pass

class GreedyAmaxDistComputer(DistComputer):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lps):
        lps = numpy.absolute(lps)
        env_id = numpy.random.choice(numpy.flatnonzero(lps == lps.max()))
        dist = self.ε*numpy.ones((len(lps)))/len(lps)
        dist[env_id] += 1-self.ε
        return dist

class PropDistComputer(DistComputer):
    ρ = 1e-8

    def __call__(self, lps):
        lps = numpy.absolute(lps) + self.ρ
        return lps/numpy.sum(lps)

class GreedyPropDistComputer(PropDistComputer):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lps):
        dist = super().__call__(lps)
        uniform = numpy.ones((len(lps)))/len(lps)
        return (1-self.ε)*dist + self.ε*uniform

class ClippedPropDistComputer(PropDistComputer):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lps):
        dist = super().__call__(lps)
        n = len(dist)
        γ = numpy.amin(dist)
        if γ < self.ε/n:
            dist = (self.ε/n - 1/n)/(γ - 1/n)*(dist - 1/n) + 1/n
        return dist

class BoltzmannDistComputer(DistComputer):
    def __init__(self, τ):
        self.τ = τ
    
    def __call__(self, lps):
        lps = numpy.absolute(lps)
        temperatured_lps = numpy.exp(lps/self.τ)
        return temperatured_lps / numpy.sum(temperatured_lps)

class GraphDistComputer(DistComputer):
    def __init__(self, G, compute_dist):
        self.G = G
        self.compute_dist = compute_dist

        mapping = {env: env_id for env_id, env in enumerate(self.G.nodes)}
        self.G = nx.relabel_nodes(self.G, mapping)

        self.focusing = numpy.array([idegree == 0 for env, idegree in G.in_degree()])

    def __call__(self, lps):
        lps = numpy.absolute(lps)

        def all_or_none(array):
            return numpy.all(array == True) or numpy.all(array == False)

        focused_env_ids = numpy.argwhere(self.focusing == True).reshape(-1)
        for env_id in focused_env_ids:
            predecessors = list(self.G.predecessors(env_id))
            successors = list(self.G.successors(env_id))
            neighboors = predecessors + successors
            self.focusing[neighboors] = lps[neighboors] > lps[env_id]
            if (numpy.any(self.focusing[predecessors + successors]) and all_or_none(self.focusing[predecessors]) and all_or_none(self.focusing[successors])):
                self.focusing[env_id] = False

        focused_env_ids = numpy.argwhere(self.focusing == True).reshape(-1)
        dist = numpy.zeros(len(lps))
        for env_id, proba in enumerate(self.compute_dist(lps[focused_env_ids])):
            predecessors = list(self.G.predecessors(env_id))
            successors = list(self.G.successors(env_id))
            family = [env_id] + predecessors + successors
            dist[family] += proba*self.compute_dist(lps[family])

        return dist