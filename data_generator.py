import torch
import numpy as np


def transform_input_sequence(X, dic=None):
    X = np.array([list(word) for word in X]).T
    if dic is None:
        dic = {str(i): i for i in range(10)}
        dic['+'] = 10

    one_hot = torch.zeros(*X.shape, len(dic))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            one_hot[i, j, dic[X[i, j]]] = 1

    return one_hot


def generate(batch_size=4096, number_of_digits=None, seq_len=None, return_one_hot=True, dic=None, seed_n=None):
    """
    :param batch_size: number of pairs (X, y) to generate
    :param seed_n: seed for generation. If None, it's set randomly.
    :param number_of_digits: number of non-zero digits in each input number (should be between 1 and seq_len, or couple)
    :param return_one_hot: should the inputs tensor (X) be one-hot-iffied ?
    :param dic: Useful to map each character/digit to a number (to define one-hot vectors). Leave to None for addition !
    :param seq_len: number of digits (zeros included) of each input number. The output will be of this length + 1
    :return: where return_one_hot is True, (X, y) where X is a tensor of shape (2 * seq_len + 1) x batch_size x  11,
    y is a tensor of shape (seq_len + 1) x batch_size. 11 comes from the fact that we encode characters 0, 1, .., 9, '+'
    when it's set to False, X and y are simple python lists of strings of type '23+45' and '68' respectively
    """

    if seed_n is None:
        seed_n = np.random.randint(0, 2 ** 32)
    try:
        rng = np.random.RandomState(seed_n)
    except ValueError:
        print(seed_n)
        return 0

    X, y = list(), list()

    fixed_length = isinstance(number_of_digits, int)

    if seq_len is None:
        seq_len = 9

    for i in range(batch_size):
        if fixed_length:
            length = number_of_digits
        else:
            length = rng.randint(number_of_digits[0], 1 + number_of_digits[1])
            # number_of_digits is a tuple specifying the min and max of sampling
        assert seq_len >= length, "seq_len={}, length={}".format(seq_len, length)

        in_pattern = [rng.randint(10 ** (length - 1), 10 ** length) for _ in range(2)]
        out_pattern = sum(in_pattern)

        in_pattern = '+'.join([str(element).zfill(seq_len) for element in in_pattern])
        out_pattern = str(out_pattern).zfill(seq_len + 1)

        X.append(in_pattern)
        y.append(out_pattern)

    if return_one_hot:
        X = transform_input_sequence(X, dic)
        y = torch.tensor(np.array([list(word) for word in y]).T.astype(int))

    return X, y




