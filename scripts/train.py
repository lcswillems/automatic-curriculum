import argparse
import time
import datetime
import torch
import torch_rl
import os
import tensorboardX

import menv
import utils
from model import ACModel

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default=None,
                    help="name of the environment to train on (REQUIRED or --graph REQUIRED)")
parser.add_argument("--graph", default=None,
                    help="name of the graph of environments to train on (REQUIRED or --env REQUIRED)")
parser.add_argument("--dist-cp", default="LpPot",
                    help="name of the distribution computer (default: LpPot)")
parser.add_argument("--lp-cp", default="Linreg",
                    help="name of the learning progress computer (default: Linreg)")
parser.add_argument("--pot-cp", default="Variable",
                    help="name of the potential computer (default: Variable)")
parser.add_argument("--dist-cr", default="GreedyProp",
                    help="name of the distribution creator (default: GreedyProp)")
parser.add_argument("--dist-alpha", type=float, default=0.1,
                    help="learning rate for TS learning progress computers (default: 0.2)")
parser.add_argument("--dist-K", type=int, default=10,
                    help="window size for some learning progress computers (default: 10)")
parser.add_argument("--dist-eps", type=float, default=0.1,
                    help="exploration coefficient for some distribution creators (default: 0.1)")
parser.add_argument("--dist-tau", type=float, default=4e-4,
                    help="temperature for Boltzmann distribution creator (default: 4e-4)")
parser.add_argument("--pot-coeff", type=float, default=0.1,
                    help="potential term coefficient in energy (default: 0.1)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: ENV_ALGO_TIME)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 128)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--gae-tau", type=float, default=0.95,
                    help="tau coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--no-instr", action="store_true", default=False,
                    help="don't use instructions in the model")
parser.add_argument("--no-mem", action="store_true", default=False,
                    help="don't use memory in the model")
args = parser.parse_args()

assert args.env is not None or args.graph is not None, "--env or --graph must be specified."

# Define model name

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_seed{}_{}".format(args.env or args.graph, args.seed, suffix)
model_name = args.model or default_model_name

# Define logger and Tensorboard writer and log script arguments

logger = utils.get_logger(model_name)
writer = tensorboardX.SummaryWriter(utils.get_log_dir(model_name))

logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments

if args.env is not None:
    envs = utils.make_envs(args.env, args.seed, args.procs)
elif args.graph is not None:
    # Load the graph and IDify it
    G = utils.load_graph(args.graph)
    G_with_ids = utils.idify_graph(G)

    # Instantiate the learning progress computer
    num_envs = len(G.nodes)
    compute_lp = {
        "Online": menv.OnlineLpComputer(num_envs, args.dist_alpha),
        "Window": menv.WindowLpComputer(num_envs, args.dist_alpha, args.dist_K),
        "Linreg": menv.LinregLpComputer(num_envs, args.dist_K),
        "None": None
    }[args.lp_cp]

    # Instantiate the potential computer
    returns = [0]*num_envs
    max_returns = [0.5]*num_envs
    compute_pot = {
        "Variable": menv.VariablePotComputer(num_envs, args.dist_K, returns, max_returns),
        "None": None
    }[args.pot_cp]

    # Instantiate the distribution creator
    create_dist = {
        "GreedyAmax": menv.GreedyAmaxDistCreator(args.dist_eps),
        "GreedyProp": menv.GreedyPropDistCreator(args.dist_eps),
        "ClippedProp": menv.ClippedPropDistCreator(args.dist_eps),
        "Boltzmann": menv.BoltzmannDistCreator(args.dist_tau),
        "None": None
    }[args.dist_cr]

    # Instantiate the distribution computer
    compute_dist = {
        "Lp": menv.LpDistComputer(compute_lp, create_dist),
        "LpPot": menv.LpPotDistComputer(compute_lp, compute_pot, create_dist, args.pot_coeff),
        "ActiveGraph": menv.ActiveGraphDistComputer(G_with_ids, compute_lp, create_dist),
        "None": None
    }[args.dist_cp]

    # Instantiate the head of the multi-environments
    head_menv = menv.HeadMultiEnv(args.procs, num_envs, compute_dist)

    # Instantiate all the multi-environments
    envs = [menv.MultiEnv(utils.make_envs_from_graph(G, args.seed + i), head_menv.remotes[i], args.seed + i)
            for i in range(args.procs)]

# Define obss preprocessor

obss_preprocessor = utils.ObssPreprocessor(model_name, envs[0].observation_space)

# Define actor-critic model

acmodel = utils.load_model(model_name, raise_not_found=False)
if acmodel is None:
    acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space, not args.no_instr, not args.no_mem)
    logger.info("Model successfully created\n")
logger.info("{}\n".format(acmodel))

if torch.cuda.is_available():
    acmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define actor-critic algo

algo = torch_rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_tau,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, obss_preprocessor,
                        utils.reshape_reward)

# Train model

num_frames = 0
total_start_time = time.time()
i = 0

while num_frames < args.frames:
    # Update parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    if args.graph is not None:
        head_menv.update_dist()
    update_end_time = time.time()
    
    num_frames += logs["num_frames"]
    i += 1

    # Print logs

    if i % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:x̄σmM {: .2f} {: .2f} {: .2f} {: .2f} | F:x̄σmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {: .3f} | vL {:.3f}"
            .format(i, num_frames, fps, duration,
                    *rreturn_per_episode.values(),
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"]))
        writer.add_scalar("frames", num_frames, i)
        writer.add_scalar("FPS", fps, i)
        writer.add_scalar("duration", total_ellapsed_time, i)
        for key, value in return_per_episode.items():
            writer.add_scalar("return_" + key, value, i)
        for key, value in rreturn_per_episode.items():
            writer.add_scalar("rreturn_" + key, value, i)
        for key, value in num_frames_per_episode.items():
            writer.add_scalar("num_frames_" + key, value, i)
        writer.add_scalar("entropy", logs["entropy"], i)
        writer.add_scalar("value", logs["value"], i)
        writer.add_scalar("policy_loss", logs["policy_loss"], i)
        writer.add_scalar("value_loss", logs["value_loss"], i)

        if args.graph is not None:
            for env_id, env_key in enumerate(G.nodes):
                writer.add_scalar("proba_{}".format(env_key),
                                  head_menv.dist[env_id], i)
                if env_id in head_menv.synthesized_returns.keys():
                    writer.add_scalar("return_{}".format(env_key),
                                        head_menv.synthesized_returns[env_id], i)
                if args.dist_cp in ["ActiveGraph"]:
                    writer.add_scalar("focus_{}".format(env_key),
                                      int(compute_dist.focusing[env_id]), i)
                if args.dist_cp in ["Lp", "ActiveGraph"]:
                    writer.add_scalar("lp_{}".format(env_key),
                                      compute_lp.lps[env_id], i)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and i % args.save_interval == 0:
        obss_preprocessor.vocab.save()

        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, model_name)
        logger.info("Model successfully saved")
        if torch.cuda.is_available():
            acmodel.cuda()