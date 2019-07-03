from abc import ABC, abstractmethod
import numpy


class LpEstimator(ABC):
    """A learning progress estimator.

    It estimates the learning progress on each environment
    from the return history."""

    def __init__(self, return_hists):
        self.return_hists = return_hists

        self.num_envs = len(self.return_hists)
        self.lps = numpy.zeros(self.num_envs)

    def __call__(self):
        for env_id in range(self.num_envs):
            self._estimate_lp(env_id)
        return self.lps

    @abstractmethod
    def _estimate_lp(self, env_id):
        pass


class TSLpEstimator(LpEstimator):
    """A learning progress estimator for Teacher-Student
    ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))
    learning progress estimators.

    It estimates an exponential moving average of the immediate
    learning progress."""

    def __init__(self, return_hists, α):
        super().__init__(return_hists)

        self.α = α

    @abstractmethod
    def _estimate_immediate_lp(self, env_id):
        pass

    def _estimate_lp(self, env_id):
        lp = self._estimate_immediate_lp(env_id)
        if lp is not None:
            self.lps[env_id] = self.α * lp + (1 - self.α) * self.lps[env_id]


class OnlineLpEstimator(TSLpEstimator):
    """The online learning progress estimator from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def _estimate_immediate_lp(self, env_id):
        _, returns = self.return_hists[env_id][-2:]
        if len(returns) >= 2:
            return returns[-1] - returns[-2]


class NaiveLpEstimator(TSLpEstimator):
    """The window learning progress estimator from the Teacher-Student
        paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def __init__(self, return_hists, α, K):
        super().__init__(return_hists, α)

        self.K = K

    def _estimate_immediate_lp(self, env_id):
        steps, returns = self.return_hists[env_id][-self.K:]
        if len(steps) >= 2:
            return numpy.polyfit(list(range(len(returns))), returns, 1)[0]


class WindowLpEstimator(TSLpEstimator):
    """The window learning progress estimator from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183))."""

    def __init__(self, return_hists, α, K):
        super().__init__(return_hists, α)

        self.K = K

    def _estimate_immediate_lp(self, env_id):
        steps, returns = self.return_hists[env_id][-self.K:]
        if len(steps) >= 2:
            return numpy.polyfit(steps, returns, 1)[0]


class SamplingLpEstimator(LpEstimator):
    """The sampling learning progress estimator from the Teacher-Student
    paper ([Matiisen et al., 2017](https://arxiv.org/abs/1707.00183)).
    Should be used with Argmax converter"""

    def __init__(self, return_hists, K):
        super().__init__(return_hists)

        self.K = K

    def _estimate_lp(self, env_id):
        steps, returns = self.return_hists[env_id][- (self.K + 1):]
        returns = numpy.array(returns)
        if len(returns) >= 2:
            self.lps[env_id] = numpy.random.choice(returns[1:] - returns[:-1])
        else:
            self.lps[env_id] = 1.


class LinregLpEstimator(LpEstimator):
    """A learning progress estimator using the immediate learning progress.

    It is similar to WindowLpEstimator except that the learning progress
    is the immediate learning progress instead of an exponential moving
    average of it."""

    def __init__(self, return_hists, K):
        super().__init__(return_hists)

        self.K = K

    def _estimate_lp(self, env_id):
        steps, returns = self.return_hists[env_id][-self.K:]
        if len(steps) >= 2:
            self.lps[env_id] = numpy.polyfit(steps, returns, 1)[0]
