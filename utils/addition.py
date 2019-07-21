import torch
import torch.nn.functional as F
import numpy as np


class AdditionAlgo:
    def __init__(self, model, optimizer, batch_size=256, num_batches=10):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_batches = num_batches

        self.num_examples = self.batch_size * self.num_batches

    def generate_data(self):
        logs = {
            "num_examples": self.num_examples,
            "accs": {}
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

            batch_pred_Y = self.model(batch_X)
            batch_loss = F.nll_loss(batch_pred_Y, batch_Y) / len(X)
            batch_acc = get_accuracy(batch_pred_Y, batch_Y)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()
            acc += batch_acc

        loss /= num_batches
        acc /= num_batches

        return {
            "loss": loss,
            "acc": acc
        }


class AdditionDataGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.char_to_one_hot_index = {
            c: i
            for i, c in enumerate(list('0123456789+'))
        }
        self.one_hot_len = len(self.char_to_one_hot_index)

    def generate(self, num_additions, num_len_probas):
        """
        :param num_len_probas: a dictionnary mapping number lengths to probas
        """

        num_lens = num_len_probas.keys()
        num_len_max = max(num_lens)
        probas = num_len_probas.values()

        X = torch.zeros(num_additions, num_len_max, self.one_hot_len)
        Y = torch.zeros(num_additions, num_len_max + 1)

        for i, num_len in enumerate(self.rng.choice(num_lens, size=num_additions, p=probas)):
            nums_to_add = [self.rng.randint(10 ** (num_len - 1), 10 ** num_len) for _ in range(2)]

            # `x` is of the form "0123+0456".
            x = '+'.join([str(num).zfill(num_len_max) for num in nums_to_add])
            for j, c in enumerate(x):
                X[i, j, self.char_to_one_hot_index[c]] = 1

            # `y` is of the form "01234".
            y = str(sum(nums_to_add)).zfill(num_len_max + 1)
            for j, c in enumerate(y):
                Y[i, j] = int(c)

        return X, Y


class FixedLenAdditionDataGenerator(AdditionDataGenerator):
    def __init__(self, num_len, seed=None):
        super().__init__(seed)

        self.num_len = num_len

    def generate(self, num_additions):
        super().generate(num_additions, {self.num_len: 1})


class DistAdditionDataGenerator(AdditionDataGenerator):
    def __init__(self, compute_dist, seed=None):
        super().__init__(seed)

        self.compute_dist = compute_dist
        self.update_dist({})

    def generate(self, num_additions):
        # TODO:
        return

    def update_dist(self, accuracies):
        # TODO:
        self.dist = self.compute_dist(accuracies)


def get_accuracy(pred_Y, Y):
    greedy_Y = pred_Y.argmax(dim=2)
    return (greedy_Y == Y).float().min(dim=0)[0].mean().item()