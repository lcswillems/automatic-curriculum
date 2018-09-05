import subprocess
import time
import itertools
import argparse

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--no-slurm", action="store_true", default=False,
                    help="don't use slurm")
args = parser.parse_args()

# Define parameters

seeds = range(1, 11)
curriculums = [
    "BlockedUnlockPickup",
    "KeyCorridor",
    "ObstructedMaze",
]
dist_cps = [
    # "Lp",
    "Learnable"
]
lp_cps = [
    "Linreg",
    # "Window",
    # "Online",
]
dist_cvs = [
    "Prop",
    # "GreedyProp",
    # "Boltzmann",
    # "GreedyAmax"
]
pot_props = [
    0,
    0.25,
    0.5,
    0.75,
    1
]

times = {
    "BlockedUnlockPickup": "0:30:0",
    "KeyCorridor": "2:0:0",
    "ObstructedMaze": "3:0:0",
}

# Execute scripts

for seed, curriculum, dist_cp, lp_cp, dist_cv, pot_prop in itertools.product(seeds, curriculums, dist_cps, lp_cps, dist_cvs, pot_props):
    slurm_cmd = "sbatch --account=def-bengioy --time={} --cpus-per-task=4 --gres=gpu:1 --mem=4G".format(times[curriculum])
    model_name = "{}_{}_{}_{}_prop{}/seed{}".format(curriculum, dist_cp, lp_cp, dist_cv, pot_prop, seed)
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --curriculum {} --dist-cp {} --lp-cp {} --dist-cr {} --pot-prop {} --model {} --save-interval 10"
        .format(slurm_cmd if not args.no_slurm else "",
                seed, curriculum, dist_cp, lp_cp, dist_cv, pot_prop, model_name),
        shell=True)
    time.sleep(1)