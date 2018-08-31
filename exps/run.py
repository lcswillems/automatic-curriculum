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
    "SC-Soft",
    # "SC-Hard",
    "BabyAI-KeyCorridor",
    "BabyAI-BlockedUnlockPickup",
    # "BabyAI-UnlockPickupDist",
    # "BabyAI-FourObjs",
    # "BabyAI-FindObj",
]
rt_hists = [
    "Normal",
    # "Gaussian"
]
dist_cps = [
    # "Lp",
    # "LpPot",
    # "LpPotRr",
    "NlpPotMancRd"
]
lp_cps = [
    "Linreg",
    # "Window",
    # "Online",
]
dist_crs = [
    "Prop",
    # "GreedyProp",
    # "Boltzmann",
    # "GreedyAmax"
]
Ks = [
    10,
    # 20,
    # 50,
    # 100
]
εs = [
    0.1,
    # 0.2,
    # 0.5,
    # 1
]
pot_props = [
    0,
    0.25,
    0.5,
    0.75,
    1
]

times = {
    "SC-Soft": "3:0:0",
    # "SC-Hard": "4:0:0",
    "BabyAI-KeyCorridor": "2:0:0",
    "BabyAI-BlockedUnlockPickup": "0:30:0",
    # "BabyAI-UnlockPickupDist": "2:0:0",
    # "BabyAI-FourObjs": "2:0:0",
    # "BabyAI-FindObj": "2:0:0"
}
no_comps = {
    "SC-Soft": "--no-instr",
    # "SC-Hard": "--no-instr",
    "BabyAI-KeyCorridor": "--no-instr",
    "BabyAI-BlockedUnlockPickup": "--no-instr",
    # "BabyAI-UnlockPickupDist": "",
    # "BabyAI-FourObjs": "",
    # "BabyAI-FindObj": "",
}

# Execute scripts

for seed, curriculum, rt_hist, dist_cp, lp_cp, dist_cr, K, ε, pot_prop in itertools.product(seeds, curriculums, rt_hists, dist_cps, lp_cps, dist_crs, Ks, εs, pot_props):
    slurm_cmd = "sbatch --account=def-bengioy --time={} --cpus-per-task=4 --gres=gpu:1 --mem=4G".format(times[curriculum])
    model_name = "{}_{}_{}_{}_{}_K{}_eps{}_pot{}/seed{}".format(curriculum, rt_hist, dist_cp, lp_cp, dist_cr, K, ε, pot_prop, seed)
    no_comp = no_comps[curriculum]
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --curriculum {} --rt-hist {} --dist-cp {} --lp-cp {} --dist-cr {} --dist-lp-K {} --dist-eps {} --pot-prop {} --model {} {} --save-interval 10"
        .format(slurm_cmd if not args.no_slurm else "",
                seed, curriculum, rt_hist, dist_cp, lp_cp, dist_cr, K, ε, pot_prop, model_name, no_comp),
        shell=True)
    time.sleep(1)