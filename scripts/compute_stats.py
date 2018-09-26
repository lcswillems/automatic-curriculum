import argparse
import glob
import os
import numpy
import pandas as pd
from subprocess import check_output
import json
import matplotlib.pyplot as plt
import re

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--stat", default=None,
                    help="select a statistic to compute")
parser.add_argument("--window", type=int, default=100,
                    help="number of time-steps to average on")
parser.add_argument("--return-to-reach", type=float, default=0.8,
                    help="return to reach, used in some statistics")
args = parser.parse_args()

# Helper functions

def load_and_clean_logs(folder, max_frame=None):
    """
    1. Load CSV logs as data frames.
    2. Clean returns, i.e. replace all NaN returns by the
       previous non NaN returns. Cache these cleaned logs.
    3. Slice data frames, i.e. keep all the rows with
       a frame number lower than `max_frame` and such that all
       data frames have the same number of rows
    """

    def clean_row(row):
        cleaned_row = numpy.zeros((len(row)))
        prev_val = 0
        for i in range(len(row)):
            if not numpy.isnan(row[i]):
                prev_val = row[i]
            cleaned_row[i] = prev_val
        return cleaned_row

    def clean_df(df):
        for col_name in df:
            if col_name.startswith("return/"):
                df[col_name] = pd.Series(clean_row(df[col_name].values), index=df.index)
        return df

    dfs = []
    model_dir = utils.get_model_dir(folder)
    pathname = os.path.join(model_dir, "**", "log.csv")
    for fname in glob.glob(pathname, recursive=True):
        clean_fname = fname + "-clean"
        if not os.path.exists(clean_fname):
            df = pd.read_csv(fname)
            df = clean_df(df)
            df.to_csv(clean_fname)
        else:
            df = pd.read_csv(clean_fname)
        dfs.append(df)
        print("{} loaded and cleaned.".format(fname))

    length = float("+inf")
    for df in dfs:
        length = min(length, df.shape[0])
        if max_frame is not None:
            length = min(length, numpy.where(df["frames"].values > max_frame)[0][0])
    for df in dfs:
        df.drop(df.index[length:], inplace=True)

    return dfs

def extract_and_smooth_df_col(dfs, col_bn):
    """
    1. Extract all the columns whose name starts with `col_bn` and group
       them by environment.
    2. Smooth all the columns with a window size given by --window.
    """

    def get_suffix_when_prefixed(s, prefix):
        if s.startswith(prefix):
            return s[len(prefix):]
        return None

    frames = dfs[0]["frames"].values
    col_data = {}
    for df in dfs:
        for col_name in df:
            env_name = get_suffix_when_prefixed(col_name, col_bn)
            if env_name is not None:
                if env_name not in col_data.keys():
                    col_data[env_name] = []
                smoothed_col = df[col_name].rolling(args.window, min_periods=1).mean().values
                col_data[env_name].append(smoothed_col)

    return {"frames": frames, col_bn: col_data}

def percentile_aggregate_data(data, percentiles):
    """
    Aggregate data by computing the percentiles for each environment.
    """

    adata = {}
    for key, matrix in data.items():
        adata[key] = {
            percentile: numpy.percentile(matrix, percentile, axis=0)
            for percentile in percentiles
        }
    return adata

def median_aggregate_data(data):
    """
    Aggregate data by averaging it for each environment.
    """

    adata = {}
    for key, matrix in data.items():
        adata[key] = numpy.median(matrix, axis=0)
    return adata

def shorten_env_name(env_name):
    """
    Remove useless parts of `env_name`.
    """

    prefix = "MiniGrid-"
    if env_name.startswith(prefix):
        env_name = env_name[len(prefix):]

    suffix = "-v0"
    if env_name.endswith(suffix):
        env_name = env_name[:-len(suffix)]

    return env_name

# Stats

stats_dir = utils.get_stats_dir()
utils.create_folders_if_necessary(stats_dir)

stat_name = "BUP-Return-GAmaxWindow-GAmaxLinreg-GPropLinreg"
if args.stat is None or args.stat == stat_name:
    # Compare the return got by Greedy Amax Window, Greedy Amax Linreg
    # and Greedy Prop Linreg on BlockedUnlockPickup curriculum.

    print(">", stat_name)

    algs = {}

    dfs = load_and_clean_logs("180923/BlockedUnlockPickup_Window_GreedyAmax_Lp_propNone", 3300000)
    algs["GAmax Window"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GAmax Window"]["areturn/"] = percentile_aggregate_data(algs["GAmax Window"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/BlockedUnlockPickup_Linreg_GreedyAmax_Lp_propNone", 3300000)
    algs["GAmax Linreg"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GAmax Linreg"]["areturn/"] = percentile_aggregate_data(algs["GAmax Linreg"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/BlockedUnlockPickup_Linreg_GreedyProp_Lp_propNone", 3300000)
    algs["GProp Linreg"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GProp Linreg"]["areturn/"] = percentile_aggregate_data(algs["GProp Linreg"]["return/"], [25, 50, 75])

    for alg, data in algs.items():
        frames = data["frames"]
        areturns = data["areturn/"]
        for env_num, env_name in enumerate(areturns.keys()):
            plt.subplot(3, 1, env_num+1)
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.title(shorten_env_name(env_name))
            plt.plot(frames, areturns[env_name][50], label=alg)
            plt.legend(loc=4, prop={'size': 6})
            plt.fill_between(
                frames,
                areturns[env_name][25],
                areturns[env_name][75],
                alpha=0.5
            )
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

stat_name = "KC-Return-GAmaxWindow-GAmaxLinreg-GPropLinreg"
if args.stat is None or args.stat == stat_name:
    # Compare the return got by Greedy Amax Window, Greedy Amax Linreg
    # and Greedy Prop Linreg on KeyCorridor curriculum.

    print(">", stat_name)

    algs = {}

    dfs = load_and_clean_logs("180923/KeyCorridor_Window_GreedyAmax_Lp_propNone", 9900000)
    algs["GAmax Window"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GAmax Window"]["areturn/"] = percentile_aggregate_data(algs["GAmax Window"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/KeyCorridor_Linreg_GreedyAmax_Lp_propNone", 9900000)
    algs["GAmax Linreg"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GAmax Linreg"]["areturn/"] = percentile_aggregate_data(algs["GAmax Linreg"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/KeyCorridor_Linreg_GreedyProp_Lp_propNone", 9900000)
    algs["GProp Linreg"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GProp Linreg"]["areturn/"] = percentile_aggregate_data(algs["GProp Linreg"]["return/"], [25, 50, 75])

    for alg, data in algs.items():
        frames = data["frames"]
        areturns = data["areturn/"]
        for env_num, env_name in enumerate(areturns.keys()):
            plt.subplot(3, 2, env_num+1)
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.title(shorten_env_name(env_name))
            plt.plot(frames, areturns[env_name][50], label=alg)
            plt.legend(loc=4 if env_num < 3 else 2, prop={'size': 6})
            plt.fill_between(
                frames,
                areturns[env_name][25],
                areturns[env_name][75],
                alpha=0.5
            )
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

stat_name = "BUP-ReturnProba-GPropLinreg"
if args.stat is None or args.stat == stat_name:
    # Display the return and the proba of Greedy Prop Linreg on
    # BlockedUnlockPickup curriculum.

    print(">", stat_name)

    dfs = load_and_clean_logs("180923/BlockedUnlockPickup_Linreg_GreedyProp_Lp_propNone", 3300000)[6:7]
    data = extract_and_smooth_df_col(dfs, "return/")
    data["areturn/"] = median_aggregate_data(data["return/"])
    data = {**data, **extract_and_smooth_df_col(dfs, "proba/")}
    data["aproba/"] = median_aggregate_data(data["proba/"])

    frames = data["frames"]
    areturns = data["areturn/"]
    for env_num, env_name in enumerate(data["areturn/"].keys()):
        plt.subplot(3, 1, env_num+1)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title(shorten_env_name(env_name))
        plt.plot(frames, data["areturn/"][env_name], label="Return")
        plt.plot(frames, data["aproba/"][env_name], label="Proba")
        plt.legend(loc="best", prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

stat_name = "OM-ReturnProba-GPropLinreg"
if args.stat is None or args.stat == stat_name:
    # Display the return and the proba of Greedy Prop Linreg on
    # ObstructedMaze curriculum.

    print(">", stat_name)

    dfs = load_and_clean_logs("180923/ObstructedMaze_Linreg_GreedyProp_Lp_propNone", 3300000)
    data = extract_and_smooth_df_col(dfs, "return/")
    data["areturn/"] = median_aggregate_data(data["return/"])
    data = {**data, **extract_and_smooth_df_col(dfs, "proba/")}
    data["aproba/"] = median_aggregate_data(data["proba/"])

    frames = data["frames"]
    areturns = data["areturn/"]
    for env_num, env_name in enumerate(data["areturn/"].keys()):
        plt.subplot(3, 3, env_num+1)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title(shorten_env_name(env_name))
        plt.plot(frames, data["areturn/"][env_name], label="Return")
        plt.plot(frames, data["aproba/"][env_name], label="Proba")
        plt.legend(loc="best", prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

stat_name = "BUP-Return-Mr-coef"
if args.stat is None or args.stat == stat_name:
    # Compare the return got by Mr algorithm with various potential coeffs
    # on BlockedUnlockPickup curriculum.

    print(">", stat_name)

    algs = {}

    dfs = load_and_clean_logs("180923/BlockedUnlockPickup_Linreg_GreedyProp_Lp_propNone", 1500000)
    algs["GProp Linreg"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GProp Linreg"]["areturn/"] = percentile_aggregate_data(algs["GProp Linreg"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/BlockedUnlockPickup_Linreg_Prop_Mr_prop0.8", 1500000)
    algs["Mr 0.8"] = extract_and_smooth_df_col(dfs, "return/")
    algs["Mr 0.8"]["areturn/"] = percentile_aggregate_data(algs["Mr 0.8"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/BlockedUnlockPickup_Linreg_Prop_Mr_prop1", 1500000)
    algs["Mr 1"] = extract_and_smooth_df_col(dfs, "return/")
    algs["Mr 1"]["areturn/"] = percentile_aggregate_data(algs["Mr 1"]["return/"], [25, 50, 75])

    for alg, data in algs.items():
        frames = data["frames"]
        areturns = data["areturn/"]
        for env_num, env_name in enumerate(areturns.keys()):
            plt.subplot(3, 1, env_num+1)
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.title(shorten_env_name(env_name))
            plt.plot(frames, areturns[env_name][50], label=alg)
            plt.legend(loc=4, prop={'size': 6})
            plt.fill_between(
                frames,
                areturns[env_name][25],
                areturns[env_name][75],
                alpha=0.5
            )
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

stat_name = "KC-Return-Mr-coef"
if args.stat is None or args.stat == stat_name:
    # Compare the return got by Mr algorithm with various potential coeffs
    # on KeyCorridor curriculum.

    print(">", stat_name)

    algs = {}

    dfs = load_and_clean_logs("180923/KeyCorridor_Linreg_GreedyProp_Lp_propNone", 9900000)
    algs["GProp Linreg"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GProp Linreg"]["areturn/"] = percentile_aggregate_data(algs["GProp Linreg"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/KeyCorridor_Linreg_Prop_Mr_prop0.8", 9900000)
    algs["Mr 0.8"] = extract_and_smooth_df_col(dfs, "return/")
    algs["Mr 0.8"]["areturn/"] = percentile_aggregate_data(algs["Mr 0.8"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/KeyCorridor_Linreg_Prop_Mr_prop1", 9900000)
    algs["Mr 1"] = extract_and_smooth_df_col(dfs, "return/")
    algs["Mr 1"]["areturn/"] = percentile_aggregate_data(algs["Mr 1"]["return/"], [25, 50, 75])

    for alg, data in algs.items():
        frames = data["frames"]
        areturns = data["areturn/"]
        for env_num, env_name in enumerate(areturns.keys()):
            plt.subplot(3, 2, env_num+1)
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.title(shorten_env_name(env_name))
            plt.plot(frames, areturns[env_name][50], label=alg)
            plt.legend(loc=4 if env_num < 3 else 2, prop={'size': 6})
            plt.fill_between(
                frames,
                areturns[env_name][25],
                areturns[env_name][75],
                alpha=0.5
            )
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

stat_name = "OM-Return-Mr-coef"
if args.stat is None or args.stat == stat_name:
    # Compare the return got by Mr algorithm with various potential coeffs
    # on ObstructedMaze curriculum.

    print(">", stat_name)

    algs = {}

    dfs = load_and_clean_logs("180923/ObstructedMaze_Linreg_GreedyProp_Lp_propNone", 9900000)
    algs["GProp Linreg"] = extract_and_smooth_df_col(dfs, "return/")
    algs["GProp Linreg"]["areturn/"] = percentile_aggregate_data(algs["GProp Linreg"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/ObstructedMaze_Linreg_Prop_Mr_prop0.8", 9900000)
    algs["Mr 0.8"] = extract_and_smooth_df_col(dfs, "return/")
    algs["Mr 0.8"]["areturn/"] = percentile_aggregate_data(algs["Mr 0.8"]["return/"], [25, 50, 75])

    dfs = load_and_clean_logs("180923/ObstructedMaze_Linreg_Prop_Mr_prop1", 9900000)
    algs["Mr 1"] = extract_and_smooth_df_col(dfs, "return/")
    algs["Mr 1"]["areturn/"] = percentile_aggregate_data(algs["Mr 1"]["return/"], [25, 50, 75])

    for alg, data in algs.items():
        frames = data["frames"]
        areturns = data["areturn/"]
        for env_num, env_name in enumerate(areturns.keys()):
            plt.subplot(3, 3, env_num+1)
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.title(shorten_env_name(env_name))
            plt.plot(frames, areturns[env_name][50], label=alg)
            plt.legend(loc=4 if env_num < 5 else 1, prop={'size': 6})
            plt.fill_between(
                frames,
                areturns[env_name][25],
                areturns[env_name][75],
                alpha=0.5
            )
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

stat_name = "OM-ReturnProba-Mr-0.8"
if args.stat is None or args.stat == stat_name:
    # Display the return and the proba of Mr with pot coeff 0.8 on
    # ObstructedMaze curriculum.

    print(">", stat_name)

    dfs = load_and_clean_logs("180923/ObstructedMaze_Linreg_Prop_Mr_prop0.8", 9900000)
    data = extract_and_smooth_df_col(dfs, "return/")
    data["areturn/"] = median_aggregate_data(data["return/"])
    data = {**data, **extract_and_smooth_df_col(dfs, "proba/")}
    data["aproba/"] = median_aggregate_data(data["proba/"])

    frames = data["frames"]
    areturns = data["areturn/"]
    for env_num, env_name in enumerate(data["areturn/"].keys()):
        plt.subplot(3, 3, env_num+1)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title(shorten_env_name(env_name))
        plt.plot(frames, data["areturn/"][env_name], label="Return")
        plt.plot(frames, data["aproba/"][env_name], label="Proba")
        plt.legend(loc="best", prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, stat_name + ".png"))
    if args.stat is not None:
        plt.show()

# stat_name = "OMFull-Return-GPropLinreg-Mr0.8"
# if args.stat is None or args.stat == stat_name:
#     print(">", stat_name)

#     def model_mean_return(model_name):
#         output = check_output(
#             "python -m scripts.evaluate --env MiniGrid-ObstructedMaze-Full-v0 --episodes 10 --model {}"
#             .format(model_name),
#             shell=True
#         ).decode("utf-8").replace("\n", "")
#         mean_return = float(re.match(".*R:x̄σmM ([0-9.]+).*", output).group(1))
#         return mean_return

#     def folder_mean_return(folder):
#         mean_returns = []

#         model_dir = utils.get_model_dir(folder)
#         pathname = os.path.join(model_dir, "**", "model.pt")
#         for fname in glob.glob(pathname, recursive=True):
#             model_name = os.path.join(*(fname.split(os.path.sep)[2:-1]))
#             mean_returns.append(model_mean_return(model_name))

#         return numpy.mean(mean_returns)

#     alg_folders = {
#         "Gprop Linreg": "180923/ObstructedMaze_Linreg_GreedyProp_Lp_propNone",
#         "Mr 0.8": "180923/ObstructedMaze_Linreg_Prop_Mr_prop0.8"
#     }
#     alg_returns = {}
#     for alg, folder in alg_folders.items():
#         alg_returns[alg] = folder_mean_return(folder)

#     txt = str(json.dumps(alg_returns, sort_keys=True, indent=4))
#     with open(os.path.join(stats_dir, stat_name+".json"), "w") as f:
#         f.write(txt)
#     if args.stat is not None:
#         print(txt)