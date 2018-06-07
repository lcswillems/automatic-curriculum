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
    "BabyAI-FourObjs",
    # "BabyAI-FindObj",
    # "BabyAI-KeyCorridor"
]
lp_cps = [
    "Linreg",
    # "Window",
    # "AbsWindow",
    # "Online",
    # "AbsOnline"
]
dist_cps = [
    "GreedyProp",
    # "ClippedProp",
    # "Boltzmann",
    # "GreedyAmax"
]
εs = [
    0.1,
    0.2,
    0.5,
    1
]
Ks = [
    10,
    20,
    50,
    100
]

times = {
    "BabyAI-BlockedUnlockPickup": "1:30:0",
    "BabyAI-UnlockPickupDist": "4:0:0",
    "BabyAI-FourObjs": "4:0:0",
    "SC-Edgeless": "4:0:0",
    "SC-Normal": "4:0:0",
    "BabyAI-FindObj": "4:0:0",
    "BabyAI-KeyCorridor": "4:0:0"
}

for seed, graph, lp_cp, dist_cp, ε, K in itertools.product(seeds, graphs, lp_cps, dist_cps, εs, Ks):
    cluster_cmd = "sbatch --account=def-bengioy --time={} --ntasks=1".format(times[graph])
    model_name = "{}_{}_{}_eps{}_K{}/seed{}".format(graph, lp_cp, dist_cp, ε, K, seed)
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --graph {} --lp {} --lp-K {} --dist {} --dist-eps {} --model {} --save-interval 10 --procs 1 --frames-per-proc 2048"
        .format(cluster_cmd if not args.no_cluster else "",
                seed, graph, lp_cp, K, dist_cp, ε, model_name),
        shell=True)
    time.sleep(1)