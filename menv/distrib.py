import numpy

class GreedyAmaxDistribComputer:
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lps):
        lps = numpy.absolute(lps)
        env_id = numpy.random.choice(numpy.flatnonzero(lps == lps.max()))
        distrib = self.ε*numpy.ones((len(lps)))/len(lps)
        distrib[env_id] += 1-self.ε
        return distrib

class PropDistribComputer:
    def __call__(self, lps):
        lps = numpy.absolute(lps)
        return lps/(numpy.sum(lps)+1e-8)

class GreedyPropDistribComputer(PropDistribComputer):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lps):
        distrib = super().__call__(lps)
        uniform = numpy.ones((len(lps)))/len(lps)
        return (1-self.ε)*distrib + self.ε*uniform

class ClippedPropDistribComputer(PropDistribComputer):
    def __init__(self, ε):
        self.ε = ε

    def __call__(self, lps):
        distrib = super().__call__(lps)
        n = len(lps)
        γ = numpy.argmin(lps)
        if γ < self.ε/n:
            distrib = (self.ε/n - 1/n)/(γ - 1/n)*(distrib - 1/n) + 1/n
        return distrib

class BoltzmannDistribComputer:
    def __init__(self, τ):
        self.τ = τ
    
    def __call__(self, lps):
        lps = numpy.absolute(lps)
        temperatured_lps = numpy.exp(lps/self.τ)
        return temperatured_lps / numpy.sum(temperatured_lps)