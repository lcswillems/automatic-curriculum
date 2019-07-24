import networkx as nx
import os
import json

import utils


def get_curriculum(curriculum_id):
    curriculum_fname = os.path.join("curriculums", curriculum_id + ".json")
    with open(curriculum_fname) as file:
        json_G = json.load(file)
        G = nx.DiGraph()
        G.add_edges_from(json_G["edges"])

    node_mapping = {}
    task_ids = []
    init_min_perfs = []
    init_max_perfs = []
    for i, node_name in enumerate(G.nodes):
        node_mapping[node_name] = i
        task_ids.append(json_G["nodes"][node_name]["id"])
        init_min_perfs.append(json_G["nodes"][node_name]["min"])
        init_max_perfs.append(json_G["nodes"][node_name]["max"])

    G = nx.relabel_nodes(G, node_mapping)

    return G, task_ids, init_min_perfs, init_max_perfs


def make_envs_from_curriculum(env_ids, seed):
    return [utils.make_env(env_id, seed) for env_id in env_ids]


def make_adds_gens_from_curriculum(gen_ids, seed):
    max_num_len = max(gen_ids)
    return [utils.make_adds_gen(num_len, max_num_len, seed) for num_len in gen_ids]
