import networkx as nx

import curriculums
import utils
import os
import json

def load_curriculum(curriculum_id):
    curriculum_fname = os.path.join("curriculums", curriculum_id + ".json")
    with open(curriculum_fname) as file:
        json_G = json.load(file)
        G = nx.DiGraph()
        G.add_edges_from(json_G["edges"])

    returns = []
    max_returns = []
    for env_key in G.nodes:
        returns.append(json_G["nodes"][env_key]["return"])
        max_returns.append(json_G["nodes"][env_key]["max_return"])

    mapping = {}
    for env_key in G.nodes:
        mapping[env_key] = json_G["nodes"][env_key]["id"]
    G = nx.relabel_nodes(G, mapping)

    return G, returns, max_returns

def make_envs_from_curriculum(G, seed):
    return [utils.make_env(env_key, seed) for env_key in G.nodes]

def idify_curriculum(G):
    mapping = {env_key: env_id for env_id, env_key in enumerate(G.nodes)}
    return nx.relabel_nodes(G, mapping)