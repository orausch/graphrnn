import networkx as nx
import numpy as np

from graphrnn_v2.stats.heuristics import MMD
from typing import Callable


def stat(heuristic: Callable[[np.ndarray, np.ndarray], float]):
    def wrapper(func: Callable[[nx.Graph], np.ndarray]):
        def wrapped(test: list[nx.Graph], pred: list[nx.Graph]):
            # Filter out empty graphs.
            test = [G for G in test if not G.number_of_nodes() == 0]
            pred = [G for G in pred if not G.number_of_nodes() == 0]

            # Compute descriptor
            test_desc = [func(G) for G in test]
            pred_desc = [func(G) for G in pred]

            # Compute heuristic
            stat = heuristic(test_desc, pred_desc)

            return stat

        return wrapped

    return wrapper


class GraphStats:
    @staticmethod
    @stat(MMD.mmd)
    def degree(G: nx.Graph) -> np.ndarray:
        return np.array(nx.degree_histogram(G))

    @staticmethod
    @stat(MMD.mmd)
    def clustering(G: nx.Graph) -> np.ndarray:
        clustering_coeffs_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(
            clustering_coeffs_list, bins=100, range=(0.0, 1.0), density=False
        )
        return hist

    @staticmethod
    @stat(MMD.mmd)
    def laplacian(G: nx.Graph) -> np.ndarray:
        eigenvals = nx.laplacian_spectrum(G)
        hist, _ = np.histogram(eigenvals, bins=100, range=(0.0, 2.0), density=False)
        return hist

    @staticmethod
    def wl(G: nx.Graph) -> np.ndarray:
        pass
