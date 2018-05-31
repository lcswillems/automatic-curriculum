import os
from subprocess import call
import sys

import envs
import utils

for senv_name in envs.get_senv_names():
    print("> Env: {}".format(senv_name))
    command = ["python -m scripts.evaluate --env {}".format(senv_name)] + sys.argv[1:]
    call(" ".join(command), shell=True)