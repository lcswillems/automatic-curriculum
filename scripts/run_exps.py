import subprocess
import time
import itertools
import argparse

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default=None,
                    help="select an experiment")
parser.add_argument("--no-slurm", action="store_true", default=False,
                    help="don't use slurm")
args = parser.parse_args()

# Define experiment running

def run_exp(curriculums, lp_ests=[None], dist_cvs=[None], dist_cps=[None], dist_cp_props=[None],
            seeds=range(1, 11), times={}):
    for curriculum, lp_est, dist_cv, dist_cp, dist_cp_prop, seed in itertools.product(curriculums, lp_ests, dist_cvs, dist_cps, dist_cp_props, seeds):
        model_name = "{}_{}_{}_{}_prop{}/seed{}".format(curriculum, lp_est, dist_cv, dist_cp, dist_cp_prop, seed)
        subprocess.Popen(
            "{} scripts/run_exps.sh python -m scripts.train {} {} {} {} {} --model {} --seed {} --save-interval 10"
            .format("sbatch --account=def-bengioy --cpus-per-task=4 --gres=gpu:1 --mem=4G --time={}".format(times[curriculum]) if not args.no_slurm else "",
                    "--curriculum {}".format(curriculum),
                    "--lp-est {}".format(lp_est) if lp_est is not None else "",
                    "--dist-cv {}".format(dist_cv) if dist_cv is not None else "",
                    "--dist-cp {}".format(dist_cp) if dist_cp is not None else "",
                    "--dist-cp-prop {}".format(dist_cp_prop) if dist_cp_prop is not None else "",
                    model_name,
                    seed),
            shell=True
        )
        time.sleep(1)

# Run experiments

if args.exp is None or args.exp == "ProofMvgAvg":
    # This shows that the moving average in the Teacher-Student
    # Window program algorithm is useless.
    run_exp(
        curriculums=[
            "BlockedUnlockPickup",
            "KeyCorridor"
        ],
        lp_ests=[
            "Window",
            "Linreg"
        ],
        dist_cvs=[
            "GreedyAmax"
        ],
        dist_cps=[
            "LP"
        ],
        seeds=range(1, 11),
        times={
            "BlockedUnlockPickup": "0:30:0",
            "KeyCorridor": "2:0:0"
        }
    )
if args.exp is None or args.exp == "ProofGreedyProp":
    # This shows that the GreedyProp attention converter leads to more
    # stability than the GreedyAmax one.
    run_exp(
        curriculums=[
            "BlockedUnlockPickup",
            "KeyCorridor",
        ],
        lp_ests=[
            "Linreg"
        ],
        dist_cvs=[
            "GreedyProp",
            "GreedyAmax"
        ],
        dist_cps=[
            "LP",
        ],
        seeds=range(1, 11),
        times={
            "BlockedUnlockPickup": "0:30:0",
            "KeyCorridor": "2:0:0",
        }
    )
if args.exp is None or args.exp == "PerfLP":
    # This gives the performance of the learning rate based program
    # algorithms.
    run_exp(
        curriculums=[
            "ObstructedMaze"
        ],
        lp_ests=[
            "Linreg",
        ],
        dist_cvs=[
            "GreedyProp",
        ],
        dist_cps=[
            "LP"
        ],
        seeds=range(1, 11),
        times={
            "ObstructedMaze": "3:0:0"
        }
    )
if args.exp is None or args.exp == "PerfMR":
    # This gives the performance of the mastering rate based program
    # algorithms.
    run_exp(
        curriculums=[
            "BlockedUnlockPickup",
            "KeyCorridor",
            "ObstructedMaze"
        ],
        lp_ests=[
            "Linreg",
        ],
        dist_cvs=[
            "Prop",
        ],
        dist_cps=[
            "MR"
        ],
        dist_cp_props=[
            0.2,
            0.4,
            0.6,
            0.8,
            1
        ],
        seeds=range(1, 11),
        times={
            "BlockedUnlockPickup": "0:30:0",
            "KeyCorridor": "2:0:0",
            "ObstructedMaze": "3:0:0"
        }
    )