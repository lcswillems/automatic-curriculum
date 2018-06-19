import subprocess
import time
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--no-cluster", action="store_true", default=False)
args = parser.parse_args()

seeds = range(1, 4)
graphs = [
    # "SC-Edgeless",
    # "SC-Normal",
    "BabyAI-BlockedUnlockPickup",
    "BabyAI-UnlockPickupDist",
    # "BabyAI-FourObjs",
    # "BabyAI-FindObj",
    # "BabyAI-KeyCorridor"
]
dist_cps = [
    # "Lp",
    "LpRwpot",
    "LpLppot",
]
lp_cps = [
    "Linreg",
    # "Window",
    # "Online",
]
rwpot_cps = [
    "Variable"
]
dist_crs = [
    "GreedyProp",
    # "ClippedProp",
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
    0.01,
    0.05,
    0.1,
    0.5,
    1
]

times = {
    "SC-D1LaIaBa": "1:30:0",
    "SC-D1LuIuBu": "3:0:0",
    "SC-D4LuIuBu": "6:0:0",
    "BabyAI-BlockedUnlockPickup": "1:30:0",
    "BabyAI-UnlockPickupDist": "4:0:0",
    "BabyAI-FourObjs": "4:0:0",
    "BabyAI-FindObj": "4:0:0",
    "BabyAI-KeyCorridor": "4:0:0"
}
no_comps = {
    "SC-D1LaIaBa": "--no-instr",
    "SC-D1LuIuBu": "--no-instr",
    "SC-D4LuIuBu": "",
    "BabyAI-BlockedUnlockPickup": "--no-instr",
    "BabyAI-UnlockPickupDist": "",
    "BabyAI-FourObjs": "",
    "BabyAI-FindObj": "",
    "BabyAI-KeyCorridor": "--no-instr"
}

for seed, graph, dist_cp, lp_cp, rwpot_cp, dist_cr, K, ε, pot_cf in itertools.product(seeds, graphs, dist_cps, lp_cps, rwpot_cps, dist_crs, Ks, εs, pot_cfs):
    cluster_cmd = "sbatch --account=def-bengioy --time={} --ntasks=1".format(times[graph])
    model_name = "{}_{}_{}_{}_K{}_eps{}_pot{}/seed{}".format(graph, dist_cp, lp_cp, dist_cr, K, ε, pot_cf, seed)
    no_comp = no_comps[graph]
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --graph {} --dist-cp {} --lp-cp {} --rwpot-cp {} --dist-cr {} --dist-K {} --dist-eps {} --pot-coeff {} --model {} {} --save-interval 10 --procs 1 --frames-per-proc 2048"
        .format(cluster_cmd if not args.no_cluster else "",
                seed, graph, dist_cp, lp_cp, rwpot_cp, dist_cr, K, ε, pot_cf, model_name, no_comp),
        shell=True)
    time.sleep(1)