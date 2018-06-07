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
dist_aut_upds = [
    False
    # True,
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

for seed, graph, lp_cp, dist_cp, dist_aut_upd, ε, K in itertools.product(seeds, graphs, lp_cps, dist_cps, dist_aut_upds, εs, Ks):
    model_name = "{}_{}_{}_{}/seed{}".format(graph, lp_cp, dist_cp, dist_aut_upd, seed)
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --graph {} --lp {} --lp-K {} --dist {} {} --dist-eps {} --model {} --save-interval 10 --procs 1 --frames-per-proc 2048"
        .format("sbatch --account=def-bengioy --time=3:0:0 --ntasks=1" if not args.no_cluster else "",
                seed, graph, lp_cp, ε, dist_cp, "--dist-automatic-update" if dist_aut_upd else "", K,
                model_name),
        shell=True)
    time.sleep(1)