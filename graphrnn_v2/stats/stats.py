import networkx as nx
import numpy as np
from graphrnn_v2.stats.heuristics import MMD


class GraphStats:
    @staticmethod
    def degree(test: list[nx.Graph], pred: list[nx.Graph]) -> float:
        # Filter out empty graphs.
        test = [G for G in test if not G.number_of_nodes() == 0]
        pred = [G for G in pred if not G.number_of_nodes() == 0]

        # Compute degree histograms.
        test_degs = [np.array(nx.degree_histogram(G)) for G in test]
        pred_degs = [np.array(nx.degree_histogram(G)) for G in pred]

        # Compute heuristic
        stat = MMD.mmd(test_degs, pred_degs)

        return stat
