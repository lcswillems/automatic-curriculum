import argparse
import csv
import networkx as nx
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model containing the CSV log (REQUIRED)")
parser.add_argument("--curriculum", required=True,
                    help="name of the curriculum from which the model was trained (REQUIRED)")
parser.add_argument("--fps", default=20,
                    help="FPS of the video (default: 20)")
args = parser.parse_args()

# Load the curriculum

G = utils.load_curriculum(args.curriculum)

# Define mapping from env names to clean env names

def clean_name(name):
    name = name.replace("-v0", "")
    name = name.replace("BabyAI-", "")
    return name.replace("SC-", "")

name_mapping = {e_name: clean_name(e_name) for e_name in G.nodes}

# Relabel nodes of curriculum

G = nx.relabel_nodes(G, name_mapping)

# Define the CSV log path

save_dir = utils.get_save_dir(args.model)
csv_path = utils.get_csv_path(save_dir)

# Define evolution paths and functions for creating / deleting
# evolution folder and getting evolution images names

evolution_folder = os.path.join(save_dir, "evolution")
evolution_video_path = os.path.join(save_dir, "evolution.mp4")

def create_evolution_folder():
    if not os.path.isdir(evolution_folder):
        os.makedirs(evolution_folder)

def delete_evolution_folder():
    shutil.rmtree(evolution_folder)

def get_evolution_image_path(id):
    return os.path.join(evolution_folder, "{}.png".format(id))

# Generate the video

H = nx.nx_agraph.to_agraph(G)
H.graph_attr["rankdir"] = "LR"
H.node_attr["shape"] = "box"
H.node_attr["style"] = "filled"
H.layout(prog="dot")

create_evolution_folder()

with open(csv_path) as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)

    # Map clean env name to the index of its corresponding env name in `header`
    proba_indexes = {
        ce_name: header.index("proba/"+e_name)
        for e_name, ce_name in name_mapping.items()
    }

    # Get the index of "frames" in `header`
    frames_index = header.index("frames")

    for data in csv_reader:
        frames = data[frames_index]

        msg = "Frames {}".format(frames)
        print(msg)

        # Create an image of the current distribution over the graph
        img_path = get_evolution_image_path(frames)
        for ce_name, index in proba_indexes.items():
            proba = float(data[index])
            red = int(255 * (1 - proba))
            node = H.get_node(ce_name)
            node.attr["fillcolor"] = "#%02x%02x%02x" % (red, 255, 255)
        H.draw(img_path)

        # Add the number of frames of training to the image
        bottom_padding = 100
        img = Image.open(img_path)
        new_size = (img.size[0], img.size[1] + bottom_padding)
        new_img = Image.new("RGB", new_size, "white")
        new_img.paste(img, (0, bottom_padding))
        draw = ImageDraw.Draw(new_img)
        draw.text((20, 20), msg, (0, 0, 0), font=ImageFont.truetype("arial.ttf", 32))
        new_img.save(img_path)

img_path = get_evolution_image_path("*")
command = "ls -v {} | xargs cat | ffmpeg -y -framerate {} -i - {}".format(img_path, args.fps, evolution_video_path)
os.system(command)

delete_evolution_folder()