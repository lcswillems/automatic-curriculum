import argparse
import time
import datetime
import torch
from torch import optim
import tensorboardX
import sys

import utils
import polyenv as penv
from model import AdditionModel


# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

## General parameters
parser.add_argument("--num-len", type=int, default=None,
                    help="number length, i.e. name of the task to train on (REQUIRED or --curriculum REQUIRED)")
parser.add_argument("--curriculum", default=None,
                    help="name of the curriculum to train on (REQUIRED or --env REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of epochs between two saves (default: 10, 0 means no saving)")
parser.add_argument("--examples", type=int, default=3 * 10**9,
                    help="number of training examples (default: 3e9) -- auto-stop if success everywhere")
parser.add_argument("--max-con-successes", type=int, default=3,
                    help="number of consecutive times models has to reach perfect accuracy to stop (default: 3)")

## Parameters for training algorithms
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--batches", type=int, default=10,
                    help="number of batches per training step (default: 10)")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate (default: 1e-3)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-beta1", type=float, default=0.9,
                    help="Adam optimizer beta1 (default: 0.9)")
parser.add_argument("--optim-beta2", type=float, default=0.999,
                    help="Adam optimizer beta2 (default: 0.999)")

## Parameters for curriculum learning algorithms
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

# TODO: review ce code
config_hash = utils.save_config(args)

# Define run dir

name = f"Addition{args.num_digits}" or args.curriculum
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{name}_seed{args.seed}_{config_hash}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Define logger, CSV writer and Tensorboard writer

logger = utils.get_logger(model_dir)
csv_file, csv_writer = utils.get_csv_writer(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))
# TODO: review ce code
logger.info("{}\n".format(config_hash))

# Set seed for all randomness sources

utils.seed(args.seed)

# Define distribution computer

if args.curriculum is not None:
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

# Load training status

try:
    status = utils.load_status(model_dir)
except OSError:
    status = {"num_examples": 0, "update": 0, "con_successes": 0}

# Define addition model

try:
    addmodel = utils.load_model(model_dir)
    logger.info("Model successfully loaded\n")
except OSError:
    addmodel = AdditionModel()
    logger.info("Model successfully created\n")
logger.info("{}\n".format(addmodel))

if torch.cuda.is_available():
    addmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define optimizer

optimizer = optim.Adam(addmodel.encoder.parameters(), args.lr,
                       (args.optim_beta1, args.optim_beta2), args.optim_eps)

try:
    utils.load_optimizer(optimizer, model_dir)
    logger.info("Optimizer successfully loaded\n")
except FileNotFoundError:
    logger.info("Optimizer successfully created\n")

# Define addition data generator

if args.num_length is not None:
    data_generator = utils.FixedLenAdditionDataGenerator(args.num_len, args.seed)
elif args.curriculum is not None:
    data_generator = utils.DistAdditionDataGenerator(compute_dist, args.seed)

# Define addition trainer

trainer = utils.AdditionTrainer(addmodel, optimizer, args.batch_size)

# Train model

num_examples = status["num_examples"]
total_start_time = time.time()
update = status["update"]
con_successes = status["con_successes"]

while num_examples < args.examples and con_successes < args.max_con_successes:
    # Update model parameters

    update_start_time = time.time()
    update_num_examples = args.batch_size * args.batches
    X, Y = data_generator.generate(update_num_examples)
    loss, accs = trainer.train(X, Y)
    if args.curriculum is not None:
        data_generator.update_dist(accs["valid_per_number"])
    update_end_time = time.time()

    num_examples += update_num_examples
    update += 1
    con_successes = con_successes + 1 if min(logs["accuracies"].values()) >= .99 else 0

    # Print logs

    if update % args.log_interval == 0 or con_successes == args.max_con_successes:
        fps = logs["num_examples"] / (update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        logger.info("U {} | F {} | FPS {} | D {}".format(update, num_examples, fps, duration))

        header = ["con_successes"]
        data = [con_successes]

        # TODO: cas env et cas curriculum

        if status["num_examples"] == 0:
            csv_writer.writerow(header)
        csv_writer.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            if value is not None:
                tb_writer.add_scalar(field, value, num_examples)

        status = {"num_examples": num_examples, "update": update, "con_successes": con_successes}
        utils.save_status(status, model_dir)

    # Save model

    if args.save_interval > 0 and (update % args.save_interval == 0 or con_successes == args.max_con_successes):
        utils.save_optimizer(optimizer, model_dir)
        logger.info("Optimizer successfully saved")
        if torch.cuda.is_available():
            addmodel.cpu()
        utils.save_model(addmodel, model_dir)
        logger.info("Model successfully saved")
        if torch.cuda.is_available():
            addmodel.cuda()
