import csv
import os
import torch
import logging
import sys
import hashlib
import pickle
from collections import OrderedDict

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "SOPH_CURRI_STORAGE" in os.environ:
        return os.environ["SOPH_CURRI_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), "models", model_name, "")


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def save_config(args):
    """
    :param args: arguments passed to a train script
    :return: a hash of those arguments to be added to the model name (and writes to the csv config what the hash means)
    """

    csv_path = os.path.join(get_storage_dir(), "config.csv")
    utils.create_folders_if_necessary(csv_path)

    try:
        csv_file = open(csv_path, "r")
    except FileNotFoundError:
        # Create the file first
        open(csv_path, "a").close()
        csv_file = open(csv_path, "r")

    reader = csv.reader(csv_file)

    args_dict = OrderedDict(sorted(vars(args).items(), key=lambda t: t[0]))

    number_of_columns = None
    for row in reader:
        number_of_columns = len(row) - 1
        break

    write_header = False
    if number_of_columns is None:
        write_header = True
    else:
        assert number_of_columns == len(args_dict.keys()), "The number of arguments changed - please use a new csv file"

    csv_file = open(csv_path, "a")
    writer = csv.writer(csv_file)

    if write_header:
        header = ['hash'] + list(args_dict.keys())
        writer.writerow(header)

    config_hash = hashlib.md5(pickle.dumps(list(args_dict.values()))).hexdigest()[:10]
    writer.writerow([config_hash] + list(args_dict.values()))

    csv_file.close()

    return config_hash
