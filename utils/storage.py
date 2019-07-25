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
    if "AUTO_CURRI_STORAGE" in os.environ:
        return os.environ["AUTO_CURRI_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


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


def save_config_in_table(args, name=None):
    """Save arguments passed to a train script in a CSV table and return a hash."""

    csv_path = os.path.join(get_storage_dir(), f"{name or 'config'}.csv")
    utils.create_folders_if_necessary(csv_path)
    if not os.path.isfile(csv_path):
        with open(csv_path, "w"): pass

    # Get current CSV header

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)

        csv_header = None
        for row in csv_reader:
            csv_header = row
            break

    # Write config (and header)

    args = OrderedDict(sorted(vars(args).items(), key=lambda t: t[0]))

    with open(csv_path, "a") as csv_file:
        csv_writer = csv.writer(csv_file)

        args_header = ["hash"] + list(args.keys())
        if csv_header is None:
            csv_writer.writerow(args_header)
        else:
            assert csv_header == args_header, "Argument names have changed. Please change the name of the current config table."

        config_hash = hashlib.md5(pickle.dumps(list(args.values()))).hexdigest()[:10]
        csv_writer.writerow([config_hash] + list(args.values()))

    return config_hash
