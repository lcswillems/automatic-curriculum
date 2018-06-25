import networkx as nx

import curriculums
import utils

def load_curriculum(curriculum_id):
    return curriculums.curriculums[curriculum_id]

def make_envs_from_curriculum(G, seed):
    return [utils.make_env(env_key, seed) for env_key in G.nodes]

def idify_curriculum(G):
    mapping = {env_key: env_id for env_id, env_key in enumerate(G.nodes)}
    return nx.relabel_nodes(G, mapping)