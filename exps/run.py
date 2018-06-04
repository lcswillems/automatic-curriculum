import subprocess
import time
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--no-cluster", action="store_true", default=False)
args = parser.parse_args()

seeds = range(1, 4)
graphs = ["SC-Edgeless", "BabyAI-BlockedUnlockPickup --no-instr --no-mem"]
lp_cps = ["Online", "AbsOnline", "Window", "AbsWindow", "AbsLinreg"]
dist_cps = ["GreedyAmax", "GreedyProp", "ClippedProp", "Boltzmann"]
dist_aut_upds = [True, False]

for seed, graph, lp_cp, dist_cp, dist_aut_upd in itertools.product(seeds, graphs, lp_cps, dist_cps, dist_aut_upds):
    model_name = "{}_{}_{}_{}/seed{}".format(graph, lp_cp, dist_cp, dist_aut_upd, seed)
    subprocess.Popen(
        "{} exps/run.sh python -m scripts.train --seed {} --graph {} --lp {} --dist {} {} --model {} --save-interval 10 --procs 1 --frames-per-proc 2048"
        .format("sbatch --account=def-bengioy --time=3:0:0 --ntasks=1" if not args.no_cluster else "",
                seed, graph, lp_cp, dist_cp, "--dist-automatic-update" if dist_aut_upd else "",
                model_name),
        shell=True)
    time.sleep(1)