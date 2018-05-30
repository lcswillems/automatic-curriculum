import os
from subprocess import call
import sys

import envs
import utils

for senv_id in envs.get_senv_ids():
    print("> Env: {}".format(senv_id))
    command = ["python -m scripts.evaluate --env {}".format(senv_id)] + sys.argv[1:]
    call(" ".join(command), shell=True)