import argparse
import torch
import os

import torch.nn as nn

from torch import optim

from model import AdditionModel
from addition_env import get_accuracy
from data_generator import generate


import utils
import polyenv as penv


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")

parser.add_argument("--seq-len", type=int, default=9,
                    help="length of each number in the input string (default: 9)")


parser.add_argument("--min-digits-valid", type=int, default=1,
                    help="if curriculum learning: evaluate on numbers of length >= this")
parser.add_argument("--max-digits-valid", type=int, default=9,
                    help="if curriculum learning: evaluate on numbers of length <= this")

parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")

parser.add_argument("--test-examples", type=int, default=100,
                    help="number of examples of each 'type' to test model on")

args = parser.parse_args()


model_name = args.model

model_dir = utils.get_model_dir(model_name)

model = utils.load_model(model_dir)


# Because sometimes, when models are trained on a cuda machine, they can't run on a non-cuda one (even after .cpu()),
# Here is an ugly hack to make things work.
# TODO: to avoid this, we should change the way models are saved and loaded in the code cleaning phase, by using state_dicts !!!!!!

state_path = os.path.join(utils.get_storage_dir(), 'test.st')
torch.save(model.state_dict(), state_path)

model = AdditionModel(output_seq_len=args.seq_len + 1)
model.load_state_dict(torch.load(state_path))


test_inputs, test_labels = zip(*(generate(args.test_examples, n_dig, args.seq_len, seed_n=args.seed)
                                 for n_dig in range(args.min_digits_valid, args.max_digits_valid + 1)))


test_results = {n_dig: get_accuracy(model(test_inputs[n_dig - args.min_digits_valid]),
                                    test_labels[n_dig - args.min_digits_valid])[1]
                            for n_dig in range(args.min_digits_valid, args.max_digits_valid + 1)}

print(test_results)
