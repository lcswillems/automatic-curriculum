import argparse
import os
from subprocess import call
import sys

import utils

# Parse arguments

args = sys.argv[1:]

try:
    i = args.index("--graph")
    graph_id = args[i+1]
    del args[i:i+2]
except ValueError:
    raise ValueError("--graph must be specified.")

# Evaluate on each environment of the graph

for env_id in utils.get_graph_env_ids(graph_id):
    print("> Env: {}".format(env_id))    
    command = ["python -m scripts.evaluate --env {}".format(env_id)] + args
    call(" ".join(command), shell=True)