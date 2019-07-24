import argparse
import time
import datetime
import torch
import torch.nn.functional as F
import tensorboardX
import sys

import utils
import auto_curri as ac
from model import AdditionModel
from sl_algo import SLAlgo


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--gen", type=int, default=None,
                    help="name of the data generator (REQUIRED or --curriculum REQUIRED)")
parser.add_argument("--curriculum", default=None,
                    help="name of the curriculum to train on (REQUIRED or --gen REQUIRED)")
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
                    help="number of examples to evaluate on a generator (default: 100)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--adam-eps", type=float, default=1e-8,
                    help="Adam optimizer epsilon (default: 1e-8)")

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

assert args.gen is not None or args.curriculum is not None, "--gen or --curriculum must be specified."

# Save the arguments in a table

config_hash = utils.save_config_in_table(args, "config_sl")

# Define run dir

name = args.gen or args.curriculum
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
txt_logger.info("Config hash: {}\n".format(config_hash))

# Set seed for all randomness sources

utils.seed(args.seed)

# Make generator

if args.gen is not None:
    gen = utils.make_gen(args.gen, args.seed)

elif args.curriculum is not None:
    # Load curriculum
    G, gen_ids, init_min_accuracies, init_max_accuracies = utils.get_curriculum(args.curriculum)

    # Make distribution computer
    compute_dist = ac.make_dist_computer(
                        len(gen_ids), args.lpe, args.lpe_alpha, args.lpe_K,
                        args.acp, G, init_min_accuracies, init_max_accuracies, args.acp_MR_K, args.acp_MR_power,
                        args.acp_MR_pot_prop, args.acp_MR_att_pred, args.acp_MR_att_succ,
                        args.a2d, args.a2d_eps, args.a2d_tau)

    # Make polymorph generator
    gen = ac.PolyGen(utils.make_gen_from_curriculum(gen_ids, args.seed), compute_dist, args.seed)

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_examples": 0, "update": 0, "con_successes": 0, "model_state": None, "optimizer_state": None}

# Define model

model = AdditionModel()
if status["model_state"] is not None:
    model.load_state_dict(status["model_state"])
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(model))

if torch.cuda.is_available():
    model.cuda()
txt_logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define supervised learning algo

criterion = lambda model_Y, Y: F.nll_loss(model_Y.transpose(1, 2), Y)
algo = SLAlgo(gen, model, criterion, args.lr, args.adam_eps,
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
    (X, Y), logs1 = algo.generate_data()
    logs2 = algo.update_parameters(X, Y)
    logs = {**logs1, **logs2}
    if args.gen is not None:
        accuracy = algo.evaluate()
        success = accuracy >= .99
    elif args.curriculum is not None:
        accuracies = algo.evaluate()
        gen.update_dist(accuracies)
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

        if args.gen is not None:
            header += ["perf"]
            data += [accuracy]
        elif args.curriculum is not None:
            for i, gen_id in enumerate(gen_ids):
                header += ["proba/{}".format(gen_id)]
                data += [gen.dist[i]]
                header += ["perf/{}".format(gen_id)]
                data += [None]
                if i in accuracies.keys():
                    data[-1] = accuracies[i]
                if args.acp in ["LP", "MR"]:
                    header += ["lp/{}".format(gen_id)]
                    data += [compute_dist.compute_att.lps[i]]
                    header += ["attention/{}".format(gen_id)]
                    data += [compute_dist.compute_att.atts[i]]
                if args.acp in ["MR"]:
                    header += ["max_perf/{}".format(gen_id)]
                    data += [compute_dist.compute_att.max_perfs[i]]
                    header += ["na_lp/{}".format(gen_id)]
                    data += [compute_dist.compute_att.na_lps[i]]
                    header += ["mr/{}".format(gen_id)]
                    data += [compute_dist.compute_att.mrs[i]]
                    header += ["anc_mr/{}".format(gen_id)]
                    data += [compute_dist.compute_att.anc_mrs[i]]
                    header += ["learning_state/{}".format(gen_id)]
                    data += [compute_dist.compute_att.learning_states[i]]
                    header += ["pre_attention/{}".format(gen_id)]
                    data += [compute_dist.compute_att.pre_atts[i]]

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
                  "model_state": model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
