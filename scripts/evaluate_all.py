import os
from subprocess import call
import sys

import utils
import envs

for env_id in envs.env_ids:
    env = "Env-{}".format(env_id)
    print("> Env: {}".format(env))
    command = ["python -m scripts.evaluate --env {}".format(env)] + sys.argv[1:]
    call(" ".join(command), shell=True)