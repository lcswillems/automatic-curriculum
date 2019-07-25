import re
import torch
import numpy


def make_gen(gen_id, seed=None):
    m = re.search("Addition([1-9]+)(m([1-9]+))?", gen_id)
    if m:
        num_len = int(m.group(1))
        max_num_len = int(m.group(3)) if m.group(3) is not None else m.group(3)
        return AdditionsGenerator(num_len, max_num_len, seed)


class AdditionsGenerator:
    char_to_one_hot_index = {c: i for i, c in enumerate(list("0123456789+"))}
    one_hot_len = 11

    def __init__(self, num_len, max_num_len=None, seed=None):
        self.num_len = num_len
        self.max_num_len = max_num_len or self.num_len
        self.rng = numpy.random.RandomState(seed)

    def generate(self, num_additions, device=None):
        X = torch.zeros(num_additions, 2 * self.max_num_len + 1, self.one_hot_len, device=device)
        Y = torch.zeros(num_additions, self.max_num_len + 1, dtype=torch.long, device=device)

        for i in range(num_additions):
            nums_to_add = [self.rng.randint(10 ** (self.num_len - 1), 10 ** self.num_len) for _ in range(2)]

            x = "+".join([str(num).zfill(self.max_num_len) for num in nums_to_add])
            for j, c in enumerate(x):
                X[i, j, self.char_to_one_hot_index[c]] = 1

            y = str(sum(nums_to_add)).zfill(self.max_num_len + 1)
            for j, c in enumerate(y):
                Y[i, j] = int(c)

        return X, Y

    def evaluate(self, model, num_additions, device=None):
        X, Y = self.generate(num_additions, device=device)
        with torch.no_grad():
            pred_Y = model(X)
        pred_Y = pred_Y.argmax(dim=2)
        return (pred_Y == Y).float().min(dim=1)[0].mean().item()
