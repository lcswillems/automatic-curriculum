import argparse
import torch

from model import AdditionModel
import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--gen", required=True,
                    help="name of the data generator (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--examples", type=int, default=1000,
                    help="number of examples of evaluation")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Make generator

gen = utils.make_gen(args.gen)

# Define model

model_dir = utils.get_model_dir(args.model)
model = AdditionModel()
model.load_state_dict(utils.get_model_state(model_dir))
print("CUDA available: {}\n".format(torch.cuda.is_available()))

# Evaluate the model

print("Accuracy: {}".format(gen.evaluate(model, args.examples)))
