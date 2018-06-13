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
        lps = self.compute_lp(returns)
        alps = numpy.absolute(lps)
        dist = self.create_dist(alps)

        return dist

class ActiveGraphDistComputer(DistComputer):
    def __init__(self, G, compute_lp, create_dist):
        self.G = G
        self.compute_lp = compute_lp
        self.create_dist = create_dist

        mapping = {env: env_id for env_id, env in enumerate(self.G.nodes)}
        self.G = nx.relabel_nodes(self.G, mapping)
        self.focusing = numpy.array([idegree == 0 for env, idegree in G.in_degree()])

    def __call__(self, returns):
        lps = self.compute_lp(returns)
        alps = numpy.absolute(lps)

        def all_or_none(array):
            return numpy.all(array == True) or numpy.all(array == False)

        focused_env_ids = numpy.argwhere(self.focusing == True).reshape(-1)
        for env_id in focused_env_ids:
            predecessors = list(self.G.predecessors(env_id))
            successors = list(self.G.successors(env_id))
            neighboors = predecessors + successors
            self.focusing[neighboors] = alps[neighboors] > alps[env_id]
            if (numpy.any(self.focusing[predecessors + successors]) and all_or_none(self.focusing[predecessors]) and all_or_none(self.focusing[successors])):
                self.focusing[env_id] = False

        focused_env_ids = numpy.argwhere(self.focusing == True).reshape(-1)
        dist = numpy.zeros(len(alps))
        for env_id, proba in enumerate(self.create_dist(alps[focused_env_ids])):
            predecessors = list(self.G.predecessors(env_id))
            successors = list(self.G.successors(env_id))
            family = [env_id] + predecessors + successors
            dist[family] += proba*self.create_dist(alps[family])

        return dist