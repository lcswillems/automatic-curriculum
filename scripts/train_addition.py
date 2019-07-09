import argparse
import time
import datetime
import torch
import tensorboardX
import sys

import torch.nn as nn

from torch import optim

from model import AdditionModel

import utils
import polyenv as penv

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
parser.add_argument("--max-digits-valid", type=int, default=None,
                    help="if curriculum learning: evaluate on numbers of length <= this")

parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")

parser.add_argument("--batchmix", action="store_true", default=False,
                    help="mix examples in the batch according to the current prob. distribution")

parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")

parser.add_argument("--examples", type=int, default=3 * 10**9,
                    help="number of training examples (default: 10e9) -- auto-stop though if success everywhere")
parser.add_argument("--max-patience", type=int, default=3,
                    help="if model reaches perfect accuracy in all validation tasks this # of times in a row, stop !")
parser.add_argument("--batch-size", type=int, default=128,
                    help="batch size (default: 128)")
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

# Generate environment

if args.num_digits is not None:
    env = utils.make_addition_env(args.seq_len, args.num_digits, args.seed)
elif args.curriculum is not None:
    # Load the curriculum, IDify it and compute the number of environments
    G, init_min_returns, init_max_returns = utils.load_curriculum(args.curriculum)
    G_with_ids = utils.idify_curriculum(G)
    num_envs = len(G.nodes)

    # Instantiate the return history for each environment
    return_hists = [penv.ReturnHistory() for _ in range(num_envs)]

    # Instantiate the learning progress estimator
    estimate_lp = {
        "Online": penv.OnlineLpEstimator(return_hists, args.lp_est_alpha),
        "Naive": penv.NaiveLpEstimator(return_hists, args.lp_est_alpha, args.lp_est_K),
        "Window": penv.WindowLpEstimator(return_hists, args.lp_est_alpha, args.lp_est_K),
        "Sampling": penv.SamplingLpEstimator(return_hists, args.lp_est_K),
        "Linreg": penv.LinregLpEstimator(return_hists, args.lp_est_K),
        "None": None
    }[args.lp_est]

    # Instantiate the distribution converter
    convert_into_dist = {
        "GreedyAmax": penv.GreedyAmaxDistConverter(args.dist_cv_eps),
        "Prop": penv.PropDistConverter(),
        "GreedyProp": penv.GreedyPropDistConverter(args.dist_cv_eps),
        "Boltzmann": penv.BoltzmannDistConverter(args.dist_cv_tau),
        "None": None
    }[args.dist_cv]

    # Instantiate the distribution computer
    compute_dist = {
        "LP": penv.LpDistComputer(return_hists, estimate_lp, convert_into_dist),
        "MR": penv.MrDistComputer(return_hists, init_min_returns, init_max_returns, args.ret_K,
                                  estimate_lp, convert_into_dist, G_with_ids, args.dist_cp_power, args.dist_cp_prop,
                                  args.dist_cp_pred_tr, args.dist_cp_succ_tr),
        "None": None
    }[args.dist_cp]

    if args.batchmix:
        envs = utils.make_mixed_addition_env_from_curriculum(G, args.seq_len, args.seed)
    else:
        envs = utils.make_addition_envs_from_curriculum(G, args.seq_len, args.seed)
    env = penv.PolySupervisedEnv(envs, args.seed, compute_dist)

else:
    raise NotImplementedError

# Load training status

try:
    status = utils.load_status(model_dir)
except OSError:
    status = {"num_examples": 0, "epoch_update": 0, "patience": 0}


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

    if args.num_digits is not None:
        results = env.train_epoch(model, encoder_optimizer, decoder_optimizer, criterion, args.epoch_length,
                                  args.batch_size, validate_using=args.valid_examples)
    elif args.curriculum is not None:
        dist = env.dist
        results = env.train_epoch(model, encoder_optimizer, decoder_optimizer, criterion, args.epoch_length,
                                  args.batch_size,
                                  (args.min_digits_valid, args.max_digits_valid or args.seq_len, args.valid_examples))
    else:
        raise NotImplementedError

    update_end_time = time.time()

    num_examples += args.batch_size * args.epoch_length
    update += 1

    loss, per_digit_ac, per_number_ac, per_digit_ac_test, per_number_ac_test, test_results = results

    # Update patience
    measure_of_success = per_number_ac_test if args.curriculum is None else min(list(zip(* test_results.values()))[1])
    patience = (patience + 1) if measure_of_success >= .99 else 0

    # Print logs

    if update % args.log_interval == 0 or patience == args.max_patience:
        fps = (args.batch_size * args.epoch_length) / (update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)

        data = [update, duration, fps, num_examples, env.number_of_digits,
                loss, 100 * per_digit_ac, 100 * per_number_ac]
        log_string = "E {}, D{}, FPS {:.1f}, Ex {}, # {}, L {:.5f}, TA: {:.3f} %, {:.3f} %, ".format(*data)

        header = ["update", "duration", "fps", "examples", "number_of_digits", "training_loss",
                  "training_accuracy_per_digit", "training_accuracy_per_operation"]

        if args.curriculum is None:
            header += ["val_accuracy_per_digit", "val_accuracy_per_operation"]
            data += [per_digit_ac_test, per_number_ac_test]
            log_string += "VA: {:.3f} %, {:.3f} %".format(100 * per_digit_ac_test, 100 * per_number_ac_test)
        else:
            header += ["val_accuracy_length_{}".format(i) for i in
                       range(args.min_digits_valid, 1 + (args.max_digits_valid or args.seq_len))]
            header += ["dist_before_epoch_env_{}".format(i) for i in range(env.num_envs)]
            new_data = [test_results[i][1] for i in range(args.min_digits_valid,
                                                          1 + (args.max_digits_valid or args.seq_len))]
            data += new_data
            data += list(dist)
            log_string += "results: " + ', '.join(["{:.3f} %".format(100 * value) for value in new_data])
            log_string += ", dist: " + ', '.join(["{:.1f} %".format(100 * value) for value in dist])

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

