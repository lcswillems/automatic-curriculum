import argparse
import glob
import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import deque

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True,
                    help="folder in storage containing CSV logs (REQUIRED)")
parser.add_argument("--curriculum", required=True,
                    help="name of the curriculum from which CSV logs are from (REQUIRED)")
parser.add_argument("--window", default=100,
                    help="number of steps to average on")
parser.add_argument("--all", action="store_true", default=False,
                    help="compute all stats")
parser.add_argument("--mean-std", action="store_true", default=False,
                    help="print and save mean standard deviation of return")
parser.add_argument("--return-line", action="store_true", default=False,
                    help="plot and save average return line")
parser.add_argument("--frame-reaching", action="store_true", default=False,
                    help="print and save the frame when some return is reached")
parser.add_argument("--frame-reaching-return", type=float, default=0.8,
                    help="return to reach in --frame-reaching")
parser.add_argument("--final-return", action="store_true", default=False,
                    help="print and save the final return")
parser.add_argument("--transfer-time-reaching", action="store_true", default=False,
                    help="print and save the time between reaching return X on task A and return X on task B when A -> B")
parser.add_argument("--transfer-time-reaching-return", type=float, default=0.3,
                    help="return to reach in --transfer-time-reaching-return")
args = parser.parse_args()

args.mean_std |= args.all
args.return_line |= args.all
args.frame_reaching |= args.all
args.final_return |= args.all
args.transfer_time_reaching |= args.all

# Load CSV logs

save_dir = utils.get_save_dir(args.folder)
pathname = os.path.join(save_dir, "**", "log.csv")
dfs = [pd.read_csv(fname) for fname in glob.glob(pathname, recursive=True)]

# Load the curriculum

G = utils.load_curriculum(args.curriculum)

# Get frames and environments smooth returns and stds for each CSV log
# Frames, smooth returns and stds are truncated such that they all have
# the same length.

def get_smooth_returns_and_stds(returns):
    sreturns = numpy.zeros(returns.shape)
    stds = numpy.zeros(returns.shape)
    last_returns = deque([], maxlen=args.window)
    for i in range(len(returns)):
        returnn = returns[i]
        if not numpy.isnan(returnn):
            last_returns.append(returnn)
        sreturns[i] = 0 if len(last_returns) == 0 else numpy.mean(last_returns)
        stds[i] = 0 if len(last_returns) == 0 else numpy.std(last_returns)
    return sreturns, stds

minlen = float("+inf")
for df in dfs:
    minlen = min(minlen, len(df["frames"]))

frames = dfs[0]["frames"].values[:minlen]
sreturnsss = []
stdsss = []
for df in dfs:
    sreturnss = {}
    stdss = {}
    for env_name in G.nodes:
        sreturns, stds = get_smooth_returns_and_stds(df["return/"+env_name].values)
        sreturnss[env_name] = sreturns[:minlen]
        stdss[env_name] = stds[:minlen]
    sreturnsss.append(sreturnss)
    stdsss.append(stdss)

# Compute --mean-std

if args.mean_std:
    print("> Mean std:")

    mean_stds = {}
    for env_name in G.nodes:
        stds_to_avg = []
        for stdss in stdsss:
            stds_to_avg.append(numpy.mean(stdss[env_name]))
        mean_stds[env_name] = {
            "mean": numpy.mean(stds_to_avg),
            "std": numpy.std(stds_to_avg)
        }

    txt = str(json.dumps(mean_stds, sort_keys=True, indent=4))
    print(txt)
    with open(os.path.join(save_dir, "stats_mean-std.json"), "w") as file:
        file.write(txt)

# Compute --return-line

if args.return_line:
    print("> Return line:")

    for env_num, env_name in enumerate(G.nodes):
        returnss_to_aggregate = []
        for sreturnss in sreturnsss:
            returnss_to_aggregate.append(sreturnss[env_name])

        plt.figure(env_num)
        plt.title(env_name)
        plt.plot(frames, numpy.mean(returnss_to_aggregate, axis=0))
        plt.fill_between(
            frames,
            numpy.amin(returnss_to_aggregate, axis=0),
            numpy.amax(returnss_to_aggregate, axis=0),
            alpha=0.5
        )
        plt.savefig(os.path.join(save_dir, "stats_return-line_{}.png".format(env_name)))
    plt.show()

# Compute --frame-reaching

if args.frame_reaching:
    print("> Frame reaching:")

    frames_reaching = {}
    for env_name in G.nodes:
        frames_to_avg = []
        for j, sreturnss in enumerate(sreturnsss):
            sreturns = sreturnss[env_name]
            i = numpy.argmax(sreturns >= args.frame_reaching_return)
            if sreturns[i] >= args.frame_reaching_return:
                frames_to_avg.append(frames[i])
        frames_reaching[env_name] = {
            "num": len(frames_to_avg),
            "mean": numpy.mean(frames_to_avg),
            "std": numpy.std(frames_to_avg)
        }

    txt = str(json.dumps(frames_reaching, sort_keys=True, indent=4))
    print(txt)
    with open(os.path.join(save_dir, "stats_frame-reaching.json"), "w") as file:
        file.write(txt)

# Compute --final-return

if args.final_return:
    print("> Final return")

    final_returns = {}
    for env_name in G.nodes:
        returns_to_avg = []
        for sreturnss in sreturnsss:
            returns_to_avg.append(sreturnss[env_name][-1])
        final_returns[env_name] = {
            "mean": numpy.mean(returns_to_avg),
            "std": numpy.std(returns_to_avg)
        }

    txt = str(json.dumps(final_returns, sort_keys=True, indent=4))
    print(txt)
    with open(os.path.join(save_dir, "stats_mean-std.json"), "w") as file:
        file.write(txt)

# Compute --transfer-time-reaching

if args.transfer_time_reaching:
    print("> Transfer time reaching")

    frames_reaching = {}
    for env_name in G.nodes:
        frames_to_avg = numpy.zeros(len(sreturnsss))
        for j, sreturnss in enumerate(sreturnsss):
            sreturns = sreturnss[env_name]
            i = numpy.argmax(sreturns >= args.transfer_time_reaching_return)
            if sreturns[i] >= args.transfer_time_reaching_return:
                frames_to_avg[j] = frames[i]
            else:
                frames_to_avg[j] = numpy.nan
        frames_reaching[env_name] = frames_to_avg

    transfer_times_reaching = {}
    for edge in G.edges:
        edge_str = " -> ".join(edge)
        transfer_time = frames_reaching[edge[1]] - frames_reaching[edge[0]]
        transfer_time = transfer_time[~numpy.isnan(transfer_time)]
        transfer_times_reaching[edge_str] = {
            "num": len(transfer_time),
            "mean": numpy.mean(transfer_time),
            "std": numpy.std(transfer_time)
        }

    txt = str(json.dumps(transfer_times_reaching, sort_keys=True, indent=4))
    print(txt)
    with open(os.path.join(save_dir, "stats_transfer-time-reaching.json"), "w") as file:
        file.write(txt)