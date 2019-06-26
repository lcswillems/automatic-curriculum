import torch
import numpy as np


def transform_input_sequence(X, dic=None, device=None):
    """
    Function that transforms strings of type "0123+0456" to the corresponding one-hot tensor, using the dictionary dic.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.array([list(word) for word in X]).T
    if dic is None:
        dic = {str(i): i for i in range(10)}
        dic['+'] = 10

    one_hot = torch.zeros(*X.shape, len(dic), device=device)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            one_hot[i, j, dic[X[i, j]]] = 1

    return one_hot


def generate(batch_size=4096, number_of_digits=None, seq_len=None, dic=None, seed_n=None, probs=None, device=None):
    """
    Function that generates pairs (x, y) where x represents a string of type "0123+0456" and y a string of type "0579".
    the x string is one-hot-encoded (11 possibilities).
    the generated pairs are grouped/split into two tensors X and y (c.f. what this function returns)
    :param batch_size: number of pairs (X, y) to generate
    :param seed_n: seed for generation. If None, it's set randomly.
    :param number_of_digits: number of non-zero digits in each input number (should be between 1 and seq_len, or couple)
    :param dic: Useful to map each character/digit to a number (to define one-hot vectors). Leave to None for addition !
    :param seq_len: number of digits (zeros included) of each input number. The output will be of this length + 1
    :param probs: when number_of_digits is a couple, this represents the ratio of examples to generate of each type.
    None is uniform.
    :return: (X, y) where X is a tensor of shape (2 * seq_len + 1) x batch_size x  11,
    y is a tensor of shape (seq_len + 1) x batch_size. 11 comes from the fact that we encode characters 0, 1, .., 9, '+'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(seed_n)

    X, y = list(), list()

    fixed_length = isinstance(number_of_digits, int)

    if seq_len is None:
        seq_len = 9

    for i in range(batch_size):
        if fixed_length:
            length = number_of_digits
        else:
            # length = rng.randint(number_of_digits[0], 1 + number_of_digits[1])
            length = rng.choice(range(number_of_digits[0], 1 + number_of_digits[1]), p=probs)
            # number_of_digits is a tuple specifying the min and max of sampling
        assert seq_len >= length, "seq_len={}, length={}".format(seq_len, length)

        # TODO: maybe generate more than one couple per iteration (but pay attention to when using probs)
        in_pattern = [rng.randint(10 ** (length - 1), 10 ** length) for _ in range(2)]
        out_pattern = sum(in_pattern)

        in_pattern = '+'.join([str(element).zfill(seq_len) for element in in_pattern])
        out_pattern = str(out_pattern).zfill(seq_len + 1)

        X.append(in_pattern)
        y.append(out_pattern)

    X = transform_input_sequence(X, dic)
    y = torch.tensor(np.array([list(word) for word in y]).T.astype(int), device=device)

    return X, y




