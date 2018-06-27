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
    "SC-Hard",
    "BabyAI-KeyCorridor"
    # "BabyAI-BlockedUnlockPickup",
    # "BabyAI-UnlockPickupDist",
    # "BabyAI-FourObjs",
    # "BabyAI-FindObj",
]
rt_hists = [
    # "Normal",
    "Gaussian"
]
dist_cps = [
    # "Lp",
    "LpPot"
]
lp_cps = [
    "Linreg",
    # "Window",
    # "Online",
]
pot_cps = [
    # "Rwpot",
    "Lppot"
]
dist_crs = [
    "GreedyProp",
    # "Boltzmann",
    # "GreedyAmax"
]
Ks = [
    # 10,
    20,
    # 50,
    # 100
]
εs = [
    # 0.1,
    0.2,
    # 0.5,
    # 1
]
pot_cfs = [
    0,
    0.001,
    0.005,
    0.01,
    # 0.05,
    # 0.1,
    # 0.5,
    # 1
]

times = {
    "SC-Soft": "6:0:0",
    "SC-Hard": "8:0:0",
    "BabyAI-BlockedUnlockPickup": "1:30:0",
    "BabyAI-UnlockPickupDist": "4:0:0",
    "BabyAI-FourObjs": "4:0:0",
    "BabyAI-FindObj": "4:0:0",
    "BabyAI-KeyCorridor": "4:0:0"
}
no_comps = {
    "SC-Soft": "--no-instr",
    "SC-Hard": "--no-instr",
    "BabyAI-BlockedUnlockPickup": "--no-instr",
    "BabyAI-UnlockPickupDist": "",
    "BabyAI-FourObjs": "",
    "BabyAI-FindObj": "",
    "BabyAI-KeyCorridor": "--no-instr"
}

# Execute scripts

for seed, curriculum, rt_hist, dist_cp, lp_cp, pot_cp, dist_cr, K, ε, pot_cf in itertools.product(seeds, curriculums, rt_hists, dist_cps, lp_cps, pot_cps, dist_crs, Ks, εs, pot_cfs):
    slurm_cmd = "sbatch --account=def-bengioy --time={} --ntasks=1".format(times[curriculum])
    model_name = "{}_{}_{}_{}_{}_K{}_eps{}_pot{}/seed{}".format(curriculum, rt_hist, dist_cp, lp_cp, dist_cr, K, ε, pot_cf, seed)
    no_comp = no_comps[curriculum]
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --curriculum {} --rt-hist {} --dist-cp {} --lp-cp {} --pot-cp {} --dist-cr {} --dist-K {} --dist-eps {} --pot-coef {} --model {} {} --save-interval 10 --procs 1 --frames-per-proc 2048"
        .format(slurm_cmd if not args.no_slurm else "",
                seed, curriculum, rt_hist, dist_cp, lp_cp, pot_cp, dist_cr, K, ε, pot_cf, model_name, no_comp),
        shell=True)
    time.sleep(1)