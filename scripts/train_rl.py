import argparse
import time
import datetime
import numpy
import torch
import torch_ac
import tensorboardX
import sys

import utils
import auto_curri as ac
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
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--adam-eps", type=float, default=1e-8,
                    help="Adam optimizer epsilon (default: 1e-8)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon (default: 0.2)")

## Parameters for curriculum learning algorithms
parser.add_argument("--lpe", default="Linreg",
                    help="name of the learning progress estimator (default: Linreg)")
parser.add_argument("--lpe-alpha", type=float, default=0.1,
                    help="learning rate for some learning progress estimators (default: 0.1)")
parser.add_argument("--lpe-K", type=int, default=10,
                    help="window size for some learning progress estimators (default: 10)")
parser.add_argument("--acp", default="MR",
                    help="name of the attention computer (default: MR)")
parser.add_argument("--acp-MR-K", type=int, default=10,
                    help="window size of the performance averaging in the MR attention computer (default: 10)")
parser.add_argument("--acp-MR-power", type=int, default=6,
                    help="power of the ancestor mastering rate for the MR attention computer (default: 6)")
parser.add_argument("--acp-MR-pot-prop", type=float, default=0.5,
                    help="potential proportion for the MR attention computer (default: 0.5)")
parser.add_argument("--acp-MR-att-pred", type=float, default=0.2,
                    help="ratio of pre-attention given to predecessors for the MR attention computer (default: 0.2)")
parser.add_argument("--acp-MR-att-succ", type=float, default=0.05,
                    help="ratio of pre-attention given to successors for the MR attention computer (default: 0.05)")
parser.add_argument("--a2d", default="Prop",
                    help="name of the attention-to-distribution converter (default: Prop)")
parser.add_argument("--a2d-eps", type=float, default=0.1,
                    help="exploration coefficient for some A2D converters (default: 0.1)")
parser.add_argument("--a2d-tau", type=float, default=4e-4,
                    help="temperature for Boltzmann A2D converter (default: 4e-4)")

args = parser.parse_args()

assert args.env is not None or args.curriculum is not None, "--env or --curriculum must be specified."

# Save the arguments in a table

config_hash = utils.save_config_in_table(args, "config_rl")

# Set run dir

name = args.env or args.curriculum
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{name}_seed{args.seed}_{config_hash}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))
txt_logger.info("Config hash: {}\n".format(config_hash))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

if args.env is not None:
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))

elif args.curriculum is not None:
    # Load curriculum
    G, env_ids, init_min_returns, init_max_returns = utils.get_curriculum(args.curriculum)

    # Make distribution computer
    compute_dist = ac.make_dist_computer(
                        len(env_ids), args.lpe, args.lpe_alpha, args.lpe_K,
                        args.acp, G, init_min_returns, init_max_returns, args.acp_MR_K, args.acp_MR_power,
                        args.acp_MR_pot_prop, args.acp_MR_att_pred, args.acp_MR_att_succ,
                        args.a2d, args.a2d_eps, args.a2d_tau)

    # Make polymorph environments
    penv_head = ac.PolyEnvHead(args.procs, len(env_ids), compute_dist)
    envs = []
    for i in range(args.procs):
        seed = args.seed + 10000 * i
        envs.append(ac.PolyEnv(utils.make_envs_from_curriculum(env_ids, seed), penv_head.remotes[i], seed))

txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0, "model_state": None, "optimizer_state": None}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
txt_logger.info("Observations preprocessor loaded")

# Load model

acmodel = ACModel(obs_space, envs[0].action_space)
if status["model_state"] is not None:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

# Load algo

algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, 1,
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
            header += ["perf"]
            data += [numpy.mean(logs["return_per_episode"])]
        elif args.curriculum is not None:
            for i, env_id in enumerate(env_ids):
                header += ["proba/{}".format(env_id)]
                data += [penv_head.dist[i]]
                header += ["perf/{}".format(env_id)]
                data += [None]
                if i in penv_head.synthesized_returns.keys():
                    data[-1] = penv_head.synthesized_returns[i]
                if args.acp in ["LP", "MR"]:
                    header += ["lp/{}".format(env_id)]
                    data += [compute_dist.compute_att.lps[i]]
                    header += ["attention/{}".format(env_id)]
                    data += [compute_dist.compute_att.atts[i]]
                if args.acp in ["MR"]:
                    header += ["max_perf/{}".format(env_id)]
                    data += [compute_dist.compute_att.max_perfs[i]]
                    header += ["na_lp/{}".format(env_id)]
                    data += [compute_dist.compute_att.na_lps[i]]
                    header += ["mr/{}".format(env_id)]
                    data += [compute_dist.compute_att.mrs[i]]
                    header += ["anc_mr/{}".format(env_id)]
                    data += [compute_dist.compute_att.anc_mrs[i]]
                    header += ["learning_state/{}".format(env_id)]
                    data += [compute_dist.compute_att.learning_states[i]]
                    header += ["pre_attention/{}".format(env_id)]
                    data += [compute_dist.compute_att.pre_atts[i]]

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
