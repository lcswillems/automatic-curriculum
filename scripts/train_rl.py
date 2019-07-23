import argparse
import time
import datetime
import numpy as np
import torch
import torch_ac
import tensorboardX
import sys

import utils
import polyenv as penv
from model import ACModel


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--env", default=None,
                    help="name of the environment to train on (REQUIRED or --curriculum REQUIRED)")
parser.add_argument("--curriculum", default=None,
                    help="name of the curriculum to train on (REQUIRED or --env REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 128)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-tau", type=float, default=0.95,
                    help="tau coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1)")
parser.add_argument("--adam-eps", type=float, default=1e-8,
                    help="Adam optimizer epsilon (default: 1e-8)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon (default: 0.2)")

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

assert args.env is not None or args.curriculum is not None, "--env or --curriculum must be specified."

# Define run dir

name = args.env or args.curriculum
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{name}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Define loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

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
        "Window": penv.WindowLpEstimator(return_hists, args.lp_est_alpha, args.lp_est_K),
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
                                  estimate_lp, convert_into_dist, G_with_ids, args.dist_cp_power, args.dist_cp_prop, args.dist_cp_pred_tr, args.dist_cp_succ_tr),
        "None": None
    }[args.dist_cp]

# Generate environments

if args.env is not None:
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
elif args.curriculum is not None:
    # Instantiate the head of the polymorph environments
    penv_head = penv.PolyEnvHead(args.procs, num_envs, compute_dist)

    # Instantiate all the polymorph environments
    envs = []
    for i in range(args.procs):
        seed = args.seed + 10000*i
        envs.append(penv.PolyEnv(utils.make_envs_from_curriculum(G, seed), penv_head.remotes[i], seed))

# Define obss preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0, "model_state": None, "optimizer_state": None}

# Define actor-critic model

acmodel = ACModel(obs_space, envs[0].action_space)
if status["model_state"] is not None:
    acmodel.load_state_dict(status["model_state"])
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

if torch.cuda.is_available():
    acmodel.cuda()
txt_logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define actor-critic algo

algo = torch_ac.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_tau,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                        args.adam_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                        utils.reshape_reward)
if status["optimizer_state"] is not None:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    if args.curriculum is not None:
        penv_head.update_dist()
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        txt_logger.info("U {} | F {} | FPS {:04.0f} | D {}".format(update, num_frames, fps, duration))

        header = []
        data = []

        if args.env is not None:
            header += ["return"]
            data += [np.mean(logs["return_per_episode"])]
        elif args.curriculum is not None:
            for env_id, env_key in enumerate(G.nodes):
                header += ["proba/{}".format(env_key)]
                data += [penv_head.dist[env_id]]
                header += ["return/{}".format(env_key)]
                data += [None]
                if env_id in penv_head.synthesized_returns.keys():
                    data[-1] = penv_head.synthesized_returns[env_id]
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

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            if value is not None:
                tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
