from abc import ABC, abstractmethod
import numpy


class DistConverter(ABC):
    """A distribution converter.

    It converts values [v_1, ..., v_N] into a distribution
    [p_1, ..., p_N]."""

    @abstractmethod
    def __call__(self, values):
        pass


class GreedyAmaxDistConverter(DistConverter):
    """A greedy argmax-based distribution converter.

    It converts values [v_1, ..., v_N] into a distribution
    [p_1, ..., p_N] where:
    - p_i = 1 - ε/N if v_i is the greatest value,
    - p_i = ε/N otherwise."""

    def __init__(self, ε):
        self.ε = ε

    def __call__(self, values):
        value_id = numpy.random.choice(numpy.flatnonzero(values == values.max()))
        dist = self.ε*numpy.ones(len(values))/len(values)
        dist[value_id] += 1-self.ε
        return dist


class PropDistConverter(DistConverter):
    """A proportionality-based distribution converter.

    It converts values [v_1, ..., v_N] into a distribution
    [p_1, ..., p_N] where p_i = v_i / (v_1 + ... + v_N)."""

    ρ = 1e-8

    def __call__(self, values):
        assert numpy.all(values >= 0), "All values must be positive."

        values = values + self.ρ
        return values/numpy.sum(values)


class GreedyPropDistConverter(PropDistConverter):
    """A greedy proportionality-based distribution converter.

    It q is the distribution converted from values [v_1, ..., v_N] by
    PropDistConverter, then it converts values into p = (1-ε)*q + ε*u
    where u is the uniform distribution."""

    def __init__(self, ε):
        self.ε = ε

    def __call__(self, values):
        dist = super().__call__(values)
        uniform = numpy.ones(len(values))/len(values)
        return (1-self.ε)*dist + self.ε*uniform


class BoltzmannDistConverter(DistConverter):
    """A Boltzmann-based distribution converter.

    It converts values [v_1, ..., v_N] into a distribution
    [p_1, ..., p_N] where p_i = exp(v_i/τ) / Σ exp(v_j/τ)."""

    def __init__(self, τ):
        self.τ = τ

    def __call__(self, values):
        temperatured_values = numpy.exp(values/self.τ)
        return temperatured_values / numpy.sum(temperatured_values)
