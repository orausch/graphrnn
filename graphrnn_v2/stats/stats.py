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

    @staticmethod
    def clustering(test: list[nx.Graph], pred: list[nx.Graph]) -> float:
        # Filter out empty graphs.
        test = [G for G in test if not G.number_of_nodes() == 0]
        pred = [G for G in pred if not G.number_of_nodes() == 0]

        # Compute clustering coefficient histograms.
        def _clustering(G: nx.Graph) -> np.array:
            clustering_coeffs_list = list(nx.clustering(G).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=100, range=(0.0, 1.0), density=False
            )
            return hist

        test_clustering = [_clustering(G) for G in test]
        pred_clustering = [_clustering(G) for G in pred]

        # Compute heuristic
        stat = MMD.mmd(test_clustering, pred_clustering)

        return stat

    @staticmethod
    def laplacian(test: list[nx.Graph], pred: list[nx.Graph]) -> float:
        pass

    @staticmethod
    def wl(test: list[nx.Graph], pred: list[nx.Graph]) -> float:
        pass
