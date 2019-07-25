import numpy
import torch


class PolyGen:
    """A polymorph data generator.

    It generates data from different data generators, mixed according
    to a distribution over data generators."""

    def __init__(self, gens, compute_dist, seed=None):
        self.gens = gens
        self.compute_dist = compute_dist
        self.rng = numpy.random.RandomState(seed)

        self._init_dist()

    def _init_dist(self):
        self.update_dist({})

    def update_dist(self, accuracies):
        self.dist = self.compute_dist(accuracies)

    def generate(self, num_examples, device=None):
        num_exampless = numpy.around(self.dist * num_examples).astype(int)

        Xs, Ys = zip(*[
            gen.generate(num_examples, device=device)
            for gen, num_examples in zip(self.gens, num_exampless)
        ])
        X, Y = torch.cat(Xs), torch.cat(Ys)

        perm = self.rng.permutation(X.size(0))

        return X[perm], Y[perm]

    def evaluate(self, model, num_examples_per_gen, device=None):
        accuracies = {}
        for i, gen in enumerate(self.gens):
            accuracies[i] = gen.evaluate(model, num_examples_per_gen, device=device)
        return accuracies
