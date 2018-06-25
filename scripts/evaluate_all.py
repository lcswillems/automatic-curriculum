import argparse
import os
from subprocess import call
import sys

import utils

# Parse arguments

args = sys.argv[1:]

try:
    i = args.index("--curriculum")
    curriculum_id = args[i+1]
    del args[i:i+2]
except ValueError:
    raise ValueError("--curriculum must be specified.")

# Evaluate on each environment of the curriculum

for env_key in utils.load_curriculum(curriculum_id).nodes:
    print("> Env: {}".format(env_key))
    command = ["python -m scripts.evaluate --env {}".format(env_key)] + args
    call(" ".join(command), shell=True)