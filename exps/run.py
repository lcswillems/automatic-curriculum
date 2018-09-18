import subprocess
import time
import itertools
import argparse

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default=None,
                    help="select experiment")
parser.add_argument("--no-slurm", action="store_true", default=False,
                    help="don't use slurm")
args = parser.parse_args()

# Define experiment running

def run_exp(curriculums, lp_ests=[None], dist_cvs=[None], dist_cps=[None], dist_cp_props=[None],
            seeds=range(1, 11), times={}):
    for curriculum, lp_est, dist_cv, dist_cp, dist_cp_prop, seed in itertools.product(curriculums, lp_ests, dist_cvs, dist_cps, dist_cp_props, seeds):
        model_name = "{}_{}_{}_{}_prop{}/seed{}".format(curriculum, lp_est, dist_cv, dist_cp, dist_cp_prop, seed)
        subprocess.Popen(
            "{} exps/run.sh python -m scripts.train {} {} {} {} {} --model {} --seed {} --save-interval 10 --no-instr"
            .format("sbatch --account=def-bengioy --cpus-per-task=4 --gres=gpu:1 --mem=4G --time={}".format(times[curriculum]) if not args.no_slurm else "",
                    "--curriculum {}".format(curriculum),
                    "--lp-cp {}".format(lp_est) if lp_est is not None else "",
                    "--dist-cv {}".format(dist_cv) if dist_cv is not None else "",
                    "--dist-cp {}".format(dist_cp) if dist_cp is not None else "",
                    "--dist-cp-prop {}".format(dist_cp_prop) if dist_cp_prop is not None else "",
                    model_name,
                    seed),
            shell=True
        )
        time.sleep(1)

# Run experiments

if args.exp is None or args.exp == "exp1":
    run_exp(
        curriculums=[
            "BlockedUnlockPickup",
            "KeyCorridor",
            "ObstructedMaze"
        ],
        lp_ests=[
            "Linreg",
            # "Window",
            # "Online"
        ],
        dist_cvs=[
            "Prop",
            # "GreedyProp",
            # "Boltzmann",
            # "GreedyAmax"
        ],
        dist_cps=[
            # "Lp",
            "Mr"
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