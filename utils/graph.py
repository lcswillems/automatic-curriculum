import networkx as nx

import graphs
import utils

def load_graph(graph_id):
    return graphs.graphs[graph_id]

def make_envs_from_graph(G, seed):
    return [utils.make_env(env_key, seed) for env_key in G.nodes]

def idify_graph(G):
    mapping = {env_key: env_id for env_id, env_key in enumerate(G.nodes)}
    return nx.relabel_nodes(G, mapping)