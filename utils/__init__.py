import os
import random
import numpy
import torch

def get_storage_dir():
    if "torch_rl_storage_dir" in os.environ:
        return os.environ["torch_rl_storage_dir"]
    return "storage"

def get_run_dir(run_name):
    return os.path.join(get_storage_dir(), run_name)

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)

def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from utils.agent import Agent
from utils.env import make_env
from utils.format import ObssPreprocessor, reshape_reward
from utils.graph import load_graph, make_envs_from_graph, idify_graph
from utils.log import synthesize, get_logger
from utils.model import model_exists, load_model, save_model, load_status, save_status
