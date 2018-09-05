import argparse
import time
import datetime
import torch
import torch_rl
import tensorboardX
import sys

import multienv as menv
import utils
from model import ACModel

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default=None,
                    help="name of the environment to train on (REQUIRED or --curriculum REQUIRED)")
parser.add_argument("--curriculum", default=None,
                    help="name of the curriculum to train on (REQUIRED or --env REQUIRED)")
parser.add_argument("--dist-cp", default="LpPotRr",
                    help="name of the distribution computer (default: LpPotRr)")
parser.add_argument("--lp-cp", default="Linreg",
                    help="name of the learning progress computer (default: Linreg)")
parser.add_argument("--dist-cr", default="GreedyProp",
                    help="name of the distribution creator (default: GreedyProp)")
parser.add_argument("--dist-alpha", type=float, default=0.1,
                    help="learning rate for TS learning progress computers (default: 0.2)")
parser.add_argument("--dist-lp-K", type=int, default=10,
                    help="window size for some learning progress computers (default: 10)")
parser.add_argument("--dist-ret-K", type=int, default=2,
                    help="window size for min and max returns (default: 2)")
parser.add_argument("--dist-eps", type=float, default=0.1,
                    help="exploration coefficient for some distribution creators (default: 0.1)")
parser.add_argument("--dist-tau", type=float, default=4e-4,
                    help="temperature for Boltzmann distribution creator (default: 4e-4)")
parser.add_argument("--dist-power", type=int, default=4,
                    help="power of the ancestor mastering rate for some distribution computer (default: 4)")
parser.add_argument("--pot-prop", type=float, default=0.5,
                    help="potential term coefficient in attention (default: 0.5)")
parser.add_argument("--tr", type=float, default=0.3,
                    help="rate of attention transfer from a node to its parents (default: 0.3)")
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

assert args.env is not None or args.curriculum is not None, "--env or --curriculum must be specified."

# Define run dir

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_seed{}_{}".format(args.env or args.curriculum, args.seed, suffix)
model_name = args.model or default_model_name
save_dir = utils.get_save_dir(model_name)

# Define logger, CSV writer and Tensorboard writer

logger = utils.get_logger(save_dir)
csv_file, csv_writer = utils.get_csv_writer(save_dir)
tb_writer = tensorboardX.SummaryWriter(save_dir)

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments

if args.env is not None:
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000*i))
elif args.curriculum is not None:
    # Load the curriculum, IDify it and compute the number of environments
    G, init_returns, init_max_returns = utils.load_curriculum(args.curriculum)
    G_with_ids = utils.idify_curriculum(G)
    num_envs = len(G.nodes)

    # Instantiate the return history for each environment
    return_hists = [menv.ReturnHistory() for _ in range(num_envs)]

    # Instantiate the learning progress computer
    compute_lp = {
        "Online": menv.OnlineLpComputer(return_hists, args.dist_alpha),
        "Window": menv.WindowLpComputer(return_hists, args.dist_alpha, args.dist_lp_K),
        "Linreg": menv.LinregLpComputer(return_hists, args.dist_lp_K),
        "None": None
    }[args.lp_cp]

    # Instantiate the distribution creator
    create_dist = {
        "GreedyAmax": menv.GreedyAmaxDistCreator(args.dist_eps),
        "Prop": menv.PropDistCreator(),
        "GreedyProp": menv.GreedyPropDistCreator(args.dist_eps),
        "Boltzmann": menv.BoltzmannDistCreator(args.dist_tau),
        "None": None
    }[args.dist_cr]

    # Instantiate the distribution computer
    compute_dist = {
        "Lp": menv.LpDistComputer(return_hists, compute_lp, create_dist),
        # "LpPot": menv.LpPotDistComputer(return_hists, init_returns, init_max_returns, args.dist_ret_K,
        #                                 compute_lp, create_dist, args.pot_coef),
        # "LpPotRr": menv.LpPotRrDistComputer(return_hists, init_returns, init_max_returns, args.dist_ret_K,
        #                                     compute_lp, create_dist, args.pot_coef, G_with_ids),
        "NlpPotMancRd": menv.NlpPotMancRdDistComputer(return_hists, init_returns, init_max_returns, args.dist_ret_K,
                                                      compute_lp, create_dist, args.pot_prop, G_with_ids, args.dist_power, args.tr),
        "None": None
    }[args.dist_cp]

    # Instantiate the head of the multi-environments
    menv_head = menv.MultiEnvHead(args.procs, num_envs, compute_dist)

    # Instantiate all the multi-environments
    envs = []
    for i in range(args.procs):
        seed = args.seed + 10000*i
        envs.append(menv.MultiEnv(utils.make_envs_from_curriculum(G, seed), menv_head.remotes[i], seed))

# Define obss preprocessor

preprocess_obss = utils.ObssPreprocessor(save_dir, envs[0].observation_space)

# Load training status

try:
    status = utils.load_status(save_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}

# Define actor-critic model

try:
    acmodel = utils.load_model(save_dir)
    logger.info("Model successfully loaded\n")
except OSError:
    acmodel = ACModel(preprocess_obss.obs_space, envs[0].action_space, not args.no_instr, not args.no_mem)
    logger.info("Model successfully created\n")
logger.info("{}\n".format(acmodel))

if torch.cuda.is_available():
    acmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define actor-critic algo

algo = torch_rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_tau,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                        utils.reshape_reward)

# Train model

num_frames = status["num_frames"]
total_start_time = time.time()
update = status["update"]

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    if args.curriculum is not None:
        menv_head.update_dist()
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        # header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        # data += rreturn_per_episode.values()
        # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        # data += num_frames_per_episode.values()
        # header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        # data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        # logger.info(
        #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
        #     .format(*data))
        logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {}"
            .format(*data))

        # header += ["return_" + key for key in return_per_episode.keys()]
        # data += return_per_episode.values()
        if args.curriculum is not None:
            for env_id, env_key in enumerate(G.nodes):
                header += ["proba/{}".format(env_key)]
                data += [menv_head.dist[env_id]]
                # header += ["return/{}".format(env_key), "return_hist/{}".format(env_key)]
                # data += [None, None]
                header += ["return/{}".format(env_key)]
                data += [None]
                if env_id in menv_head.synthesized_returns.keys():
                    data[-1] = menv_head.synthesized_returns[env_id]
                    # data[-2] = menv_head.synthesized_returns[env_id]
                    # data[-1] = compute_dist.return_hists[env_id][-1][1]
                # if args.dist_cp in ["Lp", "LpPot", "LpPotRr"]:
                #     header += ["lp/{}".format(env_key)]
                #     data += [compute_dist.lps[env_id]]
                #     header += ["attention/{}".format(env_key)]
                #     data += [compute_dist.attentions[env_id]]
                #     header += ["lp_over_attention/{}".format(env_key)]
                #     data += [None]
                #     if compute_dist.attentions[env_id] != 0:
                #         data[-1] = compute_dist.a_lps[env_id]/compute_dist.attentions[env_id]
                # if args.dist_cp in ["LpPot", "LpPotRr"]:
                #     header += ["pot/{}".format(env_key)]
                #     data += [compute_dist.pots[env_id]]
                #     header += ["maxrt/{}".format(env_key)]
                #     data += [compute_dist.max_returns[env_id]]
                # if args.dist_cp in ["LpPotRr"]:
                #     header += ["minrt/{}".format(env_key)]
                #     data += [compute_dist.min_returns[env_id]]
                #     header += ["rr/{}".format(env_key)]
                #     data += [compute_dist.rrs[env_id]]
                #     header += ["filtered_attention/{}".format(env_key)]
                #     data += [compute_dist.filtered_attentions[env_id]]
                if args.dist_cp in ["NlpPotMancRd"]:
                    header += ["maxrt/{}".format(env_key)]
                    data += [compute_dist.max_returns[env_id]]
                    header += ["lp/{}".format(env_key)]
                    data += [compute_dist.lps[env_id]]
                    header += ["na_lp/{}".format(env_key)]
                    data += [compute_dist.na_lps[env_id]]
                    header += ["mr/{}".format(env_key)]
                    data += [compute_dist.mrs[env_id]]
                    header += ["anc_mr/{}".format(env_key)]
                    data += [compute_dist.anc_mrs[env_id]]
                    header += ["learning_state/{}".format(env_key)]
                    data += [compute_dist.learning_states[env_id]]
                    header += ["attention/{}".format(env_key)]
                    data += [compute_dist.attentions[env_id]]
                    header += ["rd_attention/{}".format(env_key)]
                    data += [compute_dist.rd_attentions[env_id]]
                    header += ["na_lp_over_learning_state/{}".format(env_key)]
                    data += [None]
                    if compute_dist.learning_states[env_id] != 0:
                        data[-1] = compute_dist.na_lps[env_id]/compute_dist.learning_states[env_id]

        if status["num_frames"] == 0:
            csv_writer.writerow(header)
        csv_writer.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            if value is not None:
                tb_writer.add_scalar(field, value, num_frames)

        status = {"num_frames": num_frames, "update": update}
        utils.save_status(status, save_dir)

    # Save vocabulary and model

    if args.save_interval > 0 and update % args.save_interval == 0:
        preprocess_obss.vocab.save()

        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, save_dir)
        logger.info("Model successfully saved")
        if torch.cuda.is_available():
            acmodel.cuda()