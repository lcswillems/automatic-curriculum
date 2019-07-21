import networkx as nx
import os
import json

import utils


def load_curriculum(curriculum_id):
    curriculum_fname = os.path.join("curriculums", curriculum_id + ".json")
    with open(curriculum_fname) as file:
        json_G = json.load(file)
        G = nx.DiGraph()
        G.add_edges_from(json_G["edges"])

    init_min_returns = []
    init_max_returns = []
    for node in G.nodes:
        init_min_returns.append(json_G["nodes"][node]["min"])
        init_max_returns.append(json_G["nodes"][node]["max"])

    mapping = {}
    for node in G.nodes:
        mapping[node] = json_G["nodes"][node]["id"]
    G = nx.relabel_nodes(G, mapping)

    return G, init_min_returns, init_max_returns


def idify_curriculum(G):
    mapping = {node: id for id, node in enumerate(G.nodes)}
    return nx.relabel_nodes(G, mapping)


def make_envs_from_curriculum(G, seed):
    return [utils.make_env(env_key, seed) for env_key in G.nodes]


def make_addition_envs_from_curriculum(G, seq_len, seed):
    return [utils.make_addition_env(seq_len, env_key, seed) for env_key in G.nodes]


def make_mixed_addition_env_from_curriculum(G, seq_len, seed):
    min_len = min(G.nodes)
    max_len = max(G.nodes)
    assert sorted(G.nodes) == list(range(min_len, max_len + 1)), "graph should be contiguous"
    return utils.make_addition_env(seq_len, (min_len, max_len), seed)
