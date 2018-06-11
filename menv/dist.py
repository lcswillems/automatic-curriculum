from abc import ABC, abstractmethod
import numpy

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
    ρ = 1e-5

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