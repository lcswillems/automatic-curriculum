import numpy

class GreedyAmaxDistribComputer:
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lrs):
        lrs = numpy.absolute(lrs)
        env_id = numpy.random.choice(numpy.flatnonzero(lrs == lrs.max()))
        distrib = self.ε*numpy.ones((len(lrs)))/len(lrs)
        distrib[env_id] += 1-self.ε
        return distrib

class ProportionalDistribComputer:
    def __call__(self, lrs):
        lrs = numpy.absolute(lrs)
        return lrs/(numpy.sum(lrs)+1e-8)

class GreedyProportionalDistribComputer(ProportionalDistribComputer):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lrs):
        distrib = super().__call__(lrs)
        uniform = numpy.ones((len(lrs)))/len(lrs)
        return (1-self.ε)*distrib + self.ε*uniform

class ClippedProportionalDistribComputer(ProportionalDistribComputer):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lrs):
        distrib = super().__call__(lrs)
        n = len(lrs)
        γ = numpy.argmin(lrs)
        if γ < self.ε/n:
            distrib = (self.ε/n - 1/n)/(γ - 1/n)*(distrib - 1/n) + 1/n
        return distrib

class BoltzmannDistribComputer:
    def __init__(self, τ):
        self.τ = τ
    
    def __call__(self, lrs):
        lrs = numpy.absolute(lrs)
        temperatured_lrs = numpy.exp(lrs/self.τ)
        return temperatured_lrs / numpy.sum(temperatured_lrs)