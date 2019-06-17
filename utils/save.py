import csv
import os
import torch
import json
import logging
import sys

import utils
from collections import OrderedDict

import hashlib
import pickle


def get_model_path(model_dir):
    return os.path.join(model_dir, "model.pt")


def get_optimizers_path(model_dir):
    return os.path.join(model_dir, "optimizers.pt")


def load_model(model_dir):
    path = get_model_path(model_dir)
    model = torch.load(path)
    return model


def save_model(model, model_dir):
    path = get_model_path(model_dir)
    utils.create_folders_if_necessary(path)
    if torch.cuda.is_available():
        model.cpu()
    torch.save(model, path)
    if torch.cuda.is_available():
        model.cuda()


def save_optimizers(optimizers_dict, model_dir):
    path = get_optimizers_path(model_dir)
    state_dicts = {optimizer_name: optimizer.state_dict() for optimizer_name, optimizer in optimizers_dict.items()}
    torch.save(state_dicts, path)


def load_optimizers(optimizers_dict, model_dir):
    path = get_optimizers_path(model_dir)
    checkpoint = torch.load(path)
    for optimizer_name in optimizers_dict.keys():
        optimizers_dict[optimizer_name].load_state_dict(checkpoint[optimizer_name])


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.json")


def load_status(model_dir):
    path = get_status_path(model_dir)
    with open(path) as file:
        return json.load(file)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    with open(path, "w") as file:
        json.dump(status, file)


def get_log_path(model_dir):
    return os.path.join(model_dir, "log.txt")


def get_logger(model_dir):
    path = get_log_path(model_dir)
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


def get_csv_path(model_dir):
    return os.path.join(model_dir, "log.csv")


def get_csv_writer(model_dir):
    csv_path = get_csv_path(model_dir)
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def save_config(args):
    """
    :param args: arguments passed to a train script
    :return: a hash of those arguments to be added to the model name (and writes to the csv config what the hash means)
    """
    csv_path = utils.get_csv_config_path()
    utils.create_folders_if_necessary(csv_path)

    try:
        csv_file = open(csv_path, "r")
    except FileNotFoundError:
        # create the file first
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

    hash_value = hashlib.md5(pickle.dumps(list(args_dict.values()))).hexdigest()[:10]
    writer.writerow([hash_value] + list(args_dict.values()))

    csv_file.close()

    return hash_value
