import pathlib
import pickle
import random

import networkx as nx
import torch_geometric
from torch_geometric import data


class EgoDataset(data.InMemoryDataset):
    """M = 250"""

    def __init__(self, transform):
        super().__init__(".", transform)
        graphs = self.generate_graphs()
        graphs = [torch_geometric.utils.from_networkx(g) for g in graphs]
        self.data, self.slices = self.collate(graphs)

    @staticmethod
    def load_network():
        G = pickle.load(open(pathlib.Path(__file__).parent.parent.resolve() / "raw_datasets/ind.citeseer.graph", "rb"),
                        encoding="latin1")
        G = nx.from_dict_of_lists(G)
        G = nx.subgraph(G, max(nx.connected_components(G), key=len))
        G = nx.convert_node_labels_to_integers(G)
        return G

    @staticmethod
    def generate_graphs():
        G = EgoDataset.load_network()
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)

        return graphs


class SmallEgoDataset(EgoDataset):
    """M = 15"""

    @staticmethod
    def generate_graphs():
        """200 ego graphs with 4 ≤ |V| ≤ 20 and 3 ≤ |E| ≤ 75 (Ego-small)."""
        G = EgoDataset.load_network()
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        # FIXME: The authors shuffle the dataset before taking a subset this inherently makes reproducibility hard.
        random.shuffle(graphs)

        return graphs[0:200]
