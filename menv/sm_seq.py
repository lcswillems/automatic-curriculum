from abc import ABC, abstractmethod
import numpy

class SmoothSeq:
    def __init__(self):
        self.Xs = []
        self.Ys = []
        self.sYs = []

    def append(self, x, y):
        self.Xs.append(x)
        self.Ys.append(y)
        self.smooth_last()

    @abstractmethod
    def smooth_last(self):
        pass

    def __getitem__(self, index):
        return self.Xs[index], self.sYs[index]

class NoSmoothSeq(SmoothSeq):
    def smooth_last(self):
        self.sYs.append(self.Ys[-1])

class GaussianSmoothSeq(SmoothSeq):
    def __init__(self, σ):
        super().__init__()

        self.size = int(3*σ+1)
        self.gaussian = lambda x: 1/(σ*(2*numpy.pi)**0.5)*numpy.exp(-x**2/(2*σ**2))

    def smooth_last(self):
        Xs = numpy.array(self.Xs[-self.size:])
        Ys = numpy.array(self.Ys[-self.size:])
        unnormalized_filter = self.gaussian(Xs-Xs[-1])
        filter = unnormalized_filter / numpy.sum(unnormalized_filter)
        self.sYs.append(numpy.inner(filter, Ys))

def create_gaussian_smooth_seq():
    return GaussianSmoothSeq(10)