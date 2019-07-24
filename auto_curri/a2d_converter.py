from abc import ABC, abstractmethod
import numpy


class A2DConverter(ABC):
    """An attention-to-distribution converter.

    It converts an attention over tasks into a distribution over tasks."""

    @abstractmethod
    def __call__(self, atts):
        pass


class AmaxA2DConverter(A2DConverter):
    """The argmax attention-to-distribution converter.

    It converts an attention [a_1, ..., a_N] into a distribution
    [p_1, ..., p_N] where:
    - p_i = 1 if a_i is the greatest attention,
    - p_1 = 0 otherwise."""

    def __call__(self, atts):
        i = numpy.random.choice(numpy.flatnonzero(atts == atts.max()))
        dist = numpy.zeros(len(atts))
        dist[i] = 1
        return dist


class GreedyAmaxA2DConverter(AmaxA2DConverter):
    """The greedy-argmax attention-to-distribution converter.

    If q is the distribution obtained from an attention [a_1, ..., a_N] by
    AmaxA2DConverter, then it converts the attention into p = (1-ε)*q + ε*u
    where u is the uniform distribution."""

    def __init__(self, ε):
        self.ε = ε

    def __call__(self, atts):
        dist = super().__call__(atts)
        uniform = numpy.ones(len(atts)) / len(atts)
        return (1 - self.ε) * dist + self.ε * uniform


class PropA2DConverter(A2DConverter):
    """The proportional attention-to-distribution converter.

    It converts an attention [a_1, ..., a_N] into a distribution
    [p_1, ..., p_N] where p_i = v_i / Σ v_j."""

    ρ = 1e-8

    def __call__(self, atts):
        atts += self.ρ
        return atts / numpy.sum(atts)


class GreedyPropA2DConverter(PropA2DConverter):
    """The greedy-proportional attention-to-distribution converter.

    If q is the distribution obtained from an attention [a_1, ..., a_N] by
    PropA2DConverter, then it converts the attention into p = (1-ε)*q + ε*u
    where u is the uniform distribution."""

    def __init__(self, ε):
        self.ε = ε

    def __call__(self, atts):
        dist = super().__call__(atts)
        uniform = numpy.ones(len(atts)) / len(atts)
        return (1 - self.ε) * dist + self.ε * uniform


class BoltzmannA2DConverter(A2DConverter):
    """The Boltzmann attention-to-distribution converter.

    It converts an attention [a_1, ..., a_N] into a distribution
    [p_1, ..., p_N] where p_i = exp(v_i/τ) / Σ exp(v_j/τ)."""

    def __init__(self, τ):
        self.τ = τ

    def __call__(self, atts):
        atts = numpy.exp(atts / self.τ)
        return atts / numpy.sum(atts)
