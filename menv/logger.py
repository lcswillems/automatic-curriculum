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

        current_env_name = type(self.menv.env).__name__
        # node_labels = {}

        for env in self.menv.envs:
            env_name = type(env).__name__

            if current_env_name == env_name:
                self.writer.add_scalar(
                    "return_pe_{}".format(env_name),
                    self.menv.returnn,
                    self.num_episode)
                self.writer.add_scalar(
                    "lr_pe_{}".format(env_name),
                    self.menv.lrs[self.menv.env_id],
                    self.num_episode)
            self.writer.add_scalar(
                "proba_pe_{}".format(env_name),
                self.menv.returnn,
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