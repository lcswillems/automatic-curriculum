import argparse
import time
import datetime
import torch
import tensorboardX
import sys

import utils
import polyenv as penv
from model import AdditionModel


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--num-len", type=int, default=None,
                    help="number length (REQUIRED or --curriculum REQUIRED)")
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

## Parameters for main algorithm
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--batches", type=int, default=10,
                    help="number of batches per training step (default: 10)")
parser.add_argument("--eval-num-examples", type=int, default=100,
                    help="number of examples to evaluate on an additions generator (default: 100)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--adam-eps", type=float, default=1e-8,
                    help="Adam optimizer epsilon (default: 1e-8)")

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

assert args.num_len is not None or args.curriculum is not None, "--num-digits or --curriculum must be specified."

# Define the configuration of the arguments

# TODO: review ce code
config_hash = utils.save_config(args)

# Define run dir

name = args.curriculum or f"Addition{args.num_len}"
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{name}_seed{args.seed}_{config_hash}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Define loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))
# TODO: review ce code
txt_logger.info("Config hash: {}\n".format(config_hash))

# Set seed for all randomness sources

utils.seed(args.seed)

# Define distribution computer

if args.curriculum is not None:
    # Load the curriculum, IDify it and compute the number of environments
    G, init_min_returns, init_max_returns = utils.get_curriculum(args.curriculum)
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

# Define additions generator

if args.num_len is not None:
    adds_gen = utils.AdditionsGenerator(args.num_len, seed=args.seed)
elif args.curriculum is not None:
    adds_gen = utils.MixedAdditionsGenerator(utils.make_adds_gens_from_curriculum(G, args.seed), compute_dist, args.seed)

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_examples": 0, "update": 0, "con_successes": 0, "model_state": None, "optimizer_state": None}

# Define addition model

addmodel = AdditionModel()
if status["model_state"] is not None:
    addmodel.load_state_dict(status["model_state"])
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(addmodel))

if torch.cuda.is_available():
    addmodel.cuda()
txt_logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define addition trainer

algo = utils.AdditionAlgo(addmodel, adds_gen, args.lr, args.adam_eps,
                          args.batch_size, args.batches, args.eval_num_examples)
if status["optimizer_state"] is not None:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_examples = status["num_examples"]
update = status["update"]
con_successes = status["con_successes"]
start_time = time.time()

while num_examples < args.examples and con_successes < args.max_con_successes:
    # Update model parameters

    update_start_time = time.time()
    (X, Y), logs1 = algo.generate_additions()
    logs2 = algo.update_parameters(X, Y)
    logs = {**logs1, **logs2}
    if args.num_len is not None:
        accuracy = algo.evaluate()
        success = accuracy >= .99
    elif args.curriculum is not None:
        accuracies = algo.evaluate()
        adds_gen.update_dist(accuracies)
        success = min(accuracies.values()) >= .99
    update_end_time = time.time()

    num_examples += logs["num_examples"]
    update += 1
    con_successes = con_successes + 1 if success else 0

    # Print logs

    if update % args.log_interval == 0 or con_successes == args.max_con_successes:
        fps = logs["num_examples"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        txt_logger.info("U {} | F {} | FPS {:04.0f} | D {}".format(update, num_examples, fps, duration))

        header = ["con_successes"]
        data = [con_successes]

        if args.num_len is not None:
            header += ["accuracy"]
            data += [accuracy]
        elif args.curriculum is not None:
            for env_id, env_key in enumerate(G.nodes):
                header += ["proba/{}".format(env_key)]
                data += [adds_gen.dist[env_id]]
                header += ["return/{}".format(env_key)]
                data += [None]
                if env_id in accuracies.keys():
                    data[-1] = accuracies[env_id]
                if args.dist_cp in ["LP", "MR"]:
                    header += ["lp/{}".format(env_key)]
                    data += [compute_dist.lps[env_id]]
                    header += ["attention/{}".format(env_key)]
                    data += [compute_dist.attentions[env_id]]
                if args.dist_cp in ["MR"]:
                    header += ["maxrt/{}".format(env_key)]
                    data += [compute_dist.max_returns[env_id]]
                    header += ["na_lp/{}".format(env_key)]
                    data += [compute_dist.na_lps[env_id]]
                    header += ["mr/{}".format(env_key)]
                    data += [compute_dist.mrs[env_id]]
                    header += ["anc_mr/{}".format(env_key)]
                    data += [compute_dist.anc_mrs[env_id]]
                    header += ["learning_state/{}".format(env_key)]
                    data += [compute_dist.learning_states[env_id]]
                    header += ["pre_attention/{}".format(env_key)]
                    data += [compute_dist.pre_attentions[env_id]]

        if status["num_examples"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            if value is not None:
                tb_writer.add_scalar(field, value, num_examples)

    # Save model

    if args.save_interval > 0 and (update % args.save_interval == 0 or con_successes == args.max_con_successes):
        status = {"num_examples": num_examples, "update": update, "con_successes": con_successes,
                  "model_state": addmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
