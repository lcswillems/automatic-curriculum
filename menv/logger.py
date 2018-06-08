import os
import networkx as nx
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class MEnvLogger:
    def __init__(self, menv, writer):
        self.menv = menv
        self.writer = writer

        self.num_episode = 0

    def log(self):
        self.num_episode += 1

        # node_labels = {}

        for env_id, env in enumerate(self.menv.envs):
            env_name = type(env).__name__
            if env_id in self.menv.synthesized_returns.keys():
                self.writer.add_scalar(
                    "return_{}".format(env_name),
                    self.menv.synthesized_returns[env_id],
                    self.num_episode)
                if self.menv.lps is not None:
                    self.writer.add_scalar(
                        "lp_{}".format(env_name),
                        self.menv.lps[env_id],
                        self.num_episode)
            self.writer.add_scalar(
                "proba_{}".format(env_name),
                self.menv.dist[env_id],
                self.num_episode)
            # node_labels[env] = env_name
        
        # G = self.menv.envs_nxgraph
        # pos = nx.layout.spring_layout(G)
        # font_size = 16
        # node_sizes = [len(label) * 50*font_size for label in node_labels.values()]
        # print(node_sizes)
        # nx.draw_networkx_nodes(G, pos, node_shape="s", node_size=node_sizes)
        # nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size)
        # nx.draw_networkx_edges(G, pos, arrowstyle="->")
        # plt.axis("off")
        # plt.show()