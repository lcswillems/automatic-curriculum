from abc import ABC, abstractmethod
import numpy

class ReturnHistory:
    """The return history.

    It tracks the return given by an environment over time."""

    def __init__(self):
        self.steps = []
        self.returns = []

    def append(self, step, returnn):
        self.steps.append(step)
        self.returns.append(returnn)

    def __getitem__(self, index):
        return self.steps[index], self.returns[index]

class SmoothedReturnHistory(ReturnHistory, ABC):
    def __init__(self):
        super().__init__()

        self.smoothed_returns = []

    def append(self, step, returnn):
        super().append(step, returnn)

        self._smooth_last_return()

    @abstractmethod
    def _smooth_last_return(self):
        pass

    def __getitem__(self, index):
        return self.steps[index], self.smoothed_returns[index]

class GaussianReturnHistory(SmoothedReturnHistory):
    def __init__(self, σ):
        super().__init__()

        self.size = int(3*σ+1)
        self.gaussian = lambda x: 1/(σ*(2*numpy.pi)**0.5)*numpy.exp(-x**2/(2*σ**2))

    def _smooth_last_return(self):
        steps = numpy.array(self.steps[-self.size:])
        returns = numpy.array(self.returns[-self.size:])
        unnormalized_filter = self.gaussian(steps-steps[-1])
        filter = unnormalized_filter / numpy.sum(unnormalized_filter)
        self.smoothed_returns.append(numpy.inner(filter, returns))