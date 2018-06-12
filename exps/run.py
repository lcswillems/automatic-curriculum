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
    # "BabyAI-UnlockPickupDist",
    # "BabyAI-FourObjs",
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
Ks = [
    # 10,
    20,
    # 50,
    # 100
]
exps_graph = [
    True,
    False
]
dist_cps = [
    "GreedyProp",
    # "ClippedProp",
    # "Boltzmann",
    # "GreedyAmax"
]
εs = [
    # 0.1,
    0.2,
    # 0.5,
    # 1
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

for seed, graph, lp_cp, K, dist_cp, ε, exp_graph in itertools.product(seeds, graphs, lp_cps, Ks, dist_cps, εs, exps_graph):
    cluster_cmd = "sbatch --account=def-bengioy --time={} --ntasks=1".format(times[graph])
    model_name = "{}_{}_{}_K{}_eps{}_eg{}/seed{}".format(graph, lp_cp, dist_cp, K, ε, exp_graph, seed)
    exp_graph = "--exp-graph" if exp_graph else ""
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --graph {} --lp {} --lp-K {} --dist {} --dist-eps {} {} --model {} --save-interval 10"
        .format(cluster_cmd if not args.no_cluster else "",
                seed, graph, lp_cp, K, dist_cp, ε, exp_graph, model_name),
        shell=True)
    time.sleep(1)