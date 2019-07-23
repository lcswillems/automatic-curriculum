import torch
import torch.nn.functional as F
import numpy as np


def make_adds_gen(num_len, max_num_len=None, seed=None):
    return AdditionsGenerator(num_len, max_num_len, seed)


class AdditionsGenerator:
    char_to_one_hot_index = {c: i for i, c in enumerate(list('0123456789+'))}
    one_hot_len = 11

    def __init__(self, num_len, max_num_len=None, seed=None):
        self.num_len = num_len
        self.max_num_len = max_num_len or self.num_len
        self.rng = np.random.RandomState(seed)

    def generate(self, num_additions):
        X = torch.zeros(num_additions, 2 * self.max_num_len + 1, self.one_hot_len)
        Y = torch.zeros(num_additions, self.max_num_len + 1, dtype=torch.long)

        for i in range(num_additions):
            nums_to_add = [self.rng.randint(10 ** (self.num_len - 1), 10 ** self.num_len) for _ in range(2)]

            x = '+'.join([str(num).zfill(self.max_num_len) for num in nums_to_add])
            for j, c in enumerate(x):
                X[i, j, self.char_to_one_hot_index[c]] = 1

            y = str(sum(nums_to_add)).zfill(self.max_num_len + 1)
            for j, c in enumerate(y):
                Y[i, j] = int(c)

        return X, Y

    def evaluate(self, addmodel, num_additions):
        X, Y = self.generate(num_additions)
        return get_addition_accuracy(addmodel(X), Y)


class MixedAdditionsGenerator:
    def __init__(self, adds_gens, compute_dist, seed=None):
        self.adds_gens = adds_gens
        self.compute_dist = compute_dist
        self.rng = np.random.RandomState(seed)

        self._init_dist()

    def _init_dist(self):
        self.update_dist({})

    def update_dist(self, accuracies):
        self.dist = self.compute_dist(accuracies)

    def generate(self, num_additions):
        num_additionss = np.around(self.dist * num_additions).astype(int)

        Xs, Ys = zip(*[
            adds_gens.generate(num_additions)
            for adds_gens, num_additions in zip(self.adds_gens, num_additionss)
        ])
        X, Y = torch.cat(Xs), torch.cat(Ys)

        perm = self.rng.permutation(X.size(0))

        return X[perm], Y[perm]

    def evaluate(self, addmodel, num_additions_per_gen):
        accuracies = {}
        for i, adds_gen in enumerate(self.adds_gens):
            X, Y = adds_gen.generate(num_additions_per_gen)
            accuracies[i] = get_addition_accuracy(addmodel(X), Y)
        return accuracies


def get_addition_accuracy(pred_Y, Y):
    greedy_Y = pred_Y.argmax(dim=2)
    return (greedy_Y == Y).float().min(dim=1)[0].mean().item()


class AdditionAlgo:
    def __init__(self, addmodel, adds_gen, lr=0.001, adam_eps=1e-8,
                 batch_size=256, num_batches=10, eval_num_examples=100):
        self.addmodel = addmodel
        self.adds_gen = adds_gen
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.eval_num_examples = eval_num_examples

        self.optimizer = torch.optim.Adam(self.addmodel.parameters(), lr, eps=adam_eps)
        self.num_examples = self.batch_size * self.num_batches

    def generate_additions(self):
        X, Y = self.adds_gen.generate(self.num_examples)

        logs = {
            "num_examples": self.num_examples
        }

        return (X, Y), logs

    def update_parameters(self, X, Y):
        loss = 0
        acc = 0

        batch_start_inds = range(0, len(X), self.batch_size)
        num_batches = len(batch_start_inds)
        for i in batch_start_inds:
            batch_X = X[i:i+self.batch_size]
            batch_Y = Y[i:i+self.batch_size]

            batch_pred_Y = self.addmodel(batch_X)
            batch_loss = F.nll_loss(batch_pred_Y.transpose(1, 2), batch_Y)
            batch_acc = get_addition_accuracy(batch_pred_Y, batch_Y)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()
            acc += batch_acc

        loss /= num_batches
        acc /= num_batches

        logs = {
            "loss": loss,
            "acc": acc
        }

        return logs

    def evaluate(self):
        return self.adds_gen.evaluate(self.addmodel, self.eval_num_examples)
