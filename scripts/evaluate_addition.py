import argparse
import torch
import os

import utils
from model import AdditionModel


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--num-len", type=int, required=True,
                    help="number length (REQUIRED)")
parser.add_argument("--max-num-len", type=int, default=None,
                    help="maximum number length")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--examples", type=int, default=1000,
                    help="number of examples of evaluation")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate additions generator

adds_gen = utils.make_adds_gen(args.num_len, args.max_num_len)

# Define model

model_dir = utils.get_model_dir(args.model)
model = AdditionModel()
model.load_state_dict(utils.get_model_state(model_dir))

# Evaluate the model

X, Y = adds_gen.generate(args.examples)
print("Accuracy: {}".format(utils.get_addition_accuracy(model(X), Y)))
