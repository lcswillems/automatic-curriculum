import networkx as nx

import graphs
import utils

def make_envs_graph(graph_id, seed):
    G = graphs.graphs[graph_id]
    mapping = {env_id: utils.make_env(env_id, seed) for env_id in G.nodes}
    return nx.relabel_nodes(G, mapping)

def get_graph_env_ids(graph_id):
    G = graphs.graphs[graph_id]
    return list(G.nodes)