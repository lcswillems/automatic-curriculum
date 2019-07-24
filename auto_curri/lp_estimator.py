from abc import ABC, abstractmethod
import numpy


class LpEstimator(ABC):
    """An abstraction of a learning progress estimator.

    It estimates learning progresses on each task given performance
    histories."""

    def __init__(self, perf_hists):
        self.perf_hists = perf_hists

        self.num_tasks = len(self.perf_hists)
        self.lps = numpy.zeros(self.num_tasks)

    def __call__(self):
        for task_idx in range(self.num_tasks):
            self._estimate_lp(task_idx)
        return self.lps

    @abstractmethod
    def _estimate_lp(self, task_idx):
        pass


class EMALpEstimator(LpEstimator):
    """An abstraction of an exponential moving average learning
    progress estimator.

    It estimates an exponential moving average of the immediate
    learning progress."""

    def __init__(self, perf_hists, α):
        super().__init__(perf_hists)

        self.α = α

    @abstractmethod
    def _estimate_immediate_lp(self, task_idx):
        pass

    def _estimate_lp(self, task_idx):
        lp = self._estimate_immediate_lp(task_idx)
        if lp is not None:
            self.lps[task_idx] = self.α * lp + (1 - self.α) * self.lps[task_idx]


class OnlineLpEstimator(EMALpEstimator):
    """The online learning progress estimator from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def _estimate_immediate_lp(self, task_idx):
        _, perfs = self.perf_hists[task_idx][-2:]
        if len(perfs) >= 2:
            return perfs[-1] - perfs[-2]


class NaiveLpEstimator(EMALpEstimator):
    """The window learning progress estimator from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def __init__(self, perf_hists, α, K):
        super().__init__(perf_hists, α)

        self.K = K

    def _estimate_immediate_lp(self, task_idx):
        steps, perfs = self.perf_hists[task_idx][-self.K:]
        if len(steps) >= 2:
            return numpy.polyfit(list(range(len(perfs))), perfs, 1)[0]


class WindowLpEstimator(EMALpEstimator):
    """The window learning progress estimator from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def __init__(self, perf_hists, α, K):
        super().__init__(perf_hists, α)

        self.K = K

    def _estimate_immediate_lp(self, task_idx):
        steps, perfs = self.perf_hists[task_idx][-self.K:]
        if len(steps) >= 2:
            return numpy.polyfit(steps, perfs, 1)[0]


class SamplingLpEstimator(LpEstimator):
    """The sampling learning progress estimator from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183)).
    Must be used with the Argmax A2D converter."""

    def __init__(self, perf_hists, K):
        super().__init__(perf_hists)

        self.K = K

    def _estimate_lp(self, task_idx):
        _, perfs = self.perf_hists[task_idx][- (self.K + 1):]
        perfs = numpy.array(perfs)
        if len(perfs) >= 2:
            self.lps[task_idx] = numpy.random.choice(perfs[1:] - perfs[:-1])
        else:
            self.lps[task_idx] = 1


class LinregLpEstimator(LpEstimator):
    """The linear regression learning progress estimator.

    It is similar to WindowLpEstimator except that the learning progress
    is the immediate learning progress instead of an exponential moving
    average of it."""

    def __init__(self, perf_hists, K):
        super().__init__(perf_hists)

        self.K = K

    def _estimate_lp(self, task_idx):
        steps, perfs = self.perf_hists[task_idx][-self.K:]
        if len(steps) >= 2:
            self.lps[task_idx] = numpy.polyfit(steps, perfs, 1)[0]
