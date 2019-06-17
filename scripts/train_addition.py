import argparse
import time
import datetime
import torch
import tensorboardX
import sys

import torch.nn as nn

from torch import optim

import utils
from model import AdditionModel
from addition_env import AdditionEnvironment

from collections import OrderedDict

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--seq-len", type=int, default=9,
                    help="length of each number in the input string (default: 9)")

parser.add_argument("--num-digits", type=int, default=None,
                    help="number of non-zero digits in each input (REQUIRED or --curriculum REQUIRED)")
parser.add_argument("--curriculum", default=None,
                    help="name of the curriculum to train on (REQUIRED or --env REQUIRED)")

parser.add_argument("--valid-examples", type=int, default=None,
                    help="number of examples (of each 'type' if curriculum is not None) to validate model on")

parser.add_argument("--min-digits-valid", type=int, default=1,
                    help="if curriculum learning: evaluate on numbers of length >= this")
parser.add_argument("--max-digits-valid", type=int, default=9,
                    help="if curriculum learning: evaluate on numbers of length <= this")

parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")

parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")

parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 128)")

parser.add_argument("--examples", type=int, default=10**9,
                    help="number of training examples (default: 10e9) -- auto-stop though if success everywhere")
parser.add_argument("--max-patience", type=int, default=3,
                    help="if model reaches perfect accuracy in all validation tasks this # of times in a row, stop !")
parser.add_argument("--batch-size", type=int, default=4096,
                    help="batch size (default: 4096)")
parser.add_argument("--epoch-length", type=int, default=10,
                    help="number of batches per epoch (data generated on the fly -> this is meaningless) (default: 10)")

parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of epochs between two saves (default: 10, 0 means no saving)")

parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate (default: 1e-3)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-beta1", type=float, default=0.9,
                    help="Adam optimizer apha (default: 0.9)")
parser.add_argument("--optim-beta2", type=float, default=0.999,
                    help="Adam optimizer beta2 (default: 0.999)")

parser.add_argument("--ret-K", type=int, default=10,
                    help="window size for averaging returns (default: 10)")
parser.add_argument("--lp-est", default="Linreg",
                    help="name of the learning progress estimator (default: Linreg)")
parser.add_argument("--lp-est-alpha", type=float, default=0.1,
                    help="learning rate for TS learning progress estimators (default: 0.1)")
parser.add_argument("--lp-est-K", type=int, default=10,
                    help="window size for some learning progress estimators (default: 10)")
parser.add_argument("--dist-cv", default="Prop",
                    help="name of the distribution converter (default: Prop)")
parser.add_argument("--dist-cv-eps", type=float, default=0.1,
                    help="exploration coefficient for some distribution converters (default: 0.1)")
parser.add_argument("--dist-cv-tau", type=float, default=4e-4,
                    help="temperature for Boltzmann distribution converter (default: 4e-4)")
parser.add_argument("--dist-cp", default="MR",
                    help="name of the distribution computer (default: MR)")
parser.add_argument("--dist-cp-power", type=int, default=6,
                    help="power of the ancestor mastering rate for the MR distribution computer (default: 6)")
parser.add_argument("--dist-cp-prop", type=float, default=0.5,
                    help="potential proportion for the MR distribution computer (default: 0.5)")
parser.add_argument("--dist-cp-pred-tr", type=float, default=0.2,
                    help="attention transfer rate to predecessors for the MR distribution computer (default: 0.2)")
parser.add_argument("--dist-cp-succ-tr", type=float, default=0.05,
                    help="attention transfer rate to predecessors for the MR distribution computer (default: 0.05)")


args = parser.parse_args()

assert args.num_digits is not None or args.curriculum is not None, "--num-digits or --curriculum must be specified."

# Define the configuration of the arguments
config_hash = utils.save_config(args)

# Define run dir

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_Addition{}_seed{}_{}_{}".format(args.num_digits or args.curriculum,
                                                      args.seq_len, args.seed, config_hash, suffix)
model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Define logger, CSV writer and Tensorboard writer

logger = utils.get_logger(model_dir)
csv_file, csv_writer = utils.get_csv_writer(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))
logger.info("{}\n".format(config_hash))

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments

if args.num_digits is not None:
    envs = []
    for i in range(args.procs):
        envs.append(AdditionEnvironment(args.batch_size, args.seq_len, args.num_digits, args.seed + 10000*i))
elif args.curriculum is not None:
    # TODO
    pass

# Load training status

try:
    status = utils.load_status(model_dir)
except OSError:
    status = {"num_examples": 0, "epoch_update": 0, "patience":0}


# Define model

try:
    model = utils.load_model(model_dir)
    logger.info('Model loaded\n')
except OSError:
    model = AdditionModel(output_seq_len=args.seq_len + 1)
    logger.info('Model created\n')
logger.info("{}\n".format(model))

if torch.cuda.is_available():
    model.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))


# Define optimizers

encoder_optimizer = optim.Adam(model.encoder.parameters(), args.lr,
                               (args.optim_beta1, args.optim_beta2), args.optim_eps)
decoder_optimizer = optim.Adam(model.decoder.parameters(), args.lr,
                               (args.optim_beta1, args.optim_beta2), args.optim_eps)
optimizers = {'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer}
criterion = nn.NLLLoss()

try:
    utils.load_optimizers(optimizers, model_dir)
    logger.info("Optimizers loaded\n")
except FileNotFoundError:
    logger.info("New optimizers created\n")

# Saving model, optimizers and status
utils.save_model(model, model_dir)
utils.save_optimizers(optimizers, model_dir)
utils.save_status(status, model_dir)

# Train model

num_examples = status["num_examples"]
total_start_time = time.time()
update = status["epoch_update"]
patience = status["patience"]

while num_examples < args.examples and patience < args.max_patience:

    update_start_time = time.time()

    # TODO: multi-processing ?
    # TODO: this only considers one env case, and not polyenv
    if args.curriculum is None:
        results = envs[0].train_epoch(model, encoder_optimizer, decoder_optimizer, criterion, args.epoch_length,
                                      validate_using=args.valid_examples)
    else:
        results = envs[0].train_epoch(model, encoder_optimizer, decoder_optimizer, criterion, args.epoch_length,
                                      (args.min_digits_valid, args.max_digits_valis, args.valid_examples))

    update_end_time = time.time()

    num_examples += args.batch_size * args.epoch_length
    update += 1

    loss, (per_digit_ac, per_number_ac), (per_digit_ac_test, per_number_ac_test), test_results = results

    # Update patience
    patience = (patience + 1) if (per_number_ac_test == 1. or len(test_results) > 0 and
                                  min(test_results.values()) == 1.) else 0

    # Print logs

    if update % args.log_interval == 0 or patience == args.max_patience:
        fps = (args.batch_size * args.epoch_length) / (update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)

        log_string = "U {}, L {:.5f}, TA: {:.3f} %, {:.3f} %, ".format(update, loss, 100 * per_digit_ac,
                                                                     100 * per_number_ac)

        header = ["examples", "number_of_digits", "training_loss",
                  "training_accuracy_per_digit", "training_accuracy_per_operation"]

        data = [num_examples, envs[0].number_of_digits, loss, per_digit_ac, per_number_ac]

        if args.curriculum is None:
            header += ["val_accuracy_per_digit", "val_accuracy_per_operation"]
            data += [per_digit_ac_test, per_number_ac_test]
            log_string += "VA: {:.3f} %, {:.3f} %".format(100 * per_digit_ac_test, 100 * per_number_ac_test)
        else:
            header += ["val_accuracy_length_{}".format(i) for i in range(args.min_digits_valid, args.max_digits_valis)]
            new_data = [test_results[i][1] for i in range(args.min_digits_valid, args.max_digits_valid)]
            data += new_data
            log_string += "results: " + ', '.join(["{:.3f} %".format(100 * value) for value in new_data])

        header += ['patience']
        data += [patience]

        log_string += ", patience {}".format(patience)
        logger.info(log_string)

        if status["num_examples"] == 0:
            csv_writer.writerow(header)
        csv_writer.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            if value is not None:
                tb_writer.add_scalar(field, value, num_examples)

        status = {"num_examples": num_examples, "epoch_update": update, "patience": patience}
        utils.save_status(status, model_dir)

    # Save model

    if args.save_interval > 0 and (update % args.save_interval == 0 or patience == args.max_patience):

        utils.save_model(model, model_dir)
        logger.info("Model successfully saved")

        utils.save_optimizers(optimizers, model_dir)
        logger.info("Optimizers successfully saved")

        # TODO: note that if training interrupts before a model is saved (i.e. not right just after it, mismatch will happen between status and model)
