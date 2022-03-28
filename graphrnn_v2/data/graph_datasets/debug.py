import networkx as nx
import torch_geometric
from torch_geometric import data


class DebugDataset(data.InMemoryDataset):
    """
    Creates a dataset containing only the following debug graph

    (0)---(1)
     |     |
    (2)---(3)
      \   /
       (4)

    """

    def __init__(self, transform):
        super().__init__(".", transform)
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
        graphs = [torch_geometric.utils.from_networkx(self.G)]
        self.data, self.slices = self.collate(graphs)


class MixedDebugDataset(data.InMemoryDataset):
    """
    Creates a dataset containing the following two debug graphs.
    That is to check how graphs of different sizes aggregate.

    (0)---(1)               (0)---(1)
     |     |                 |
    (2)---(3)               (2)
      \   /
       (4)

    """

    def __init__(self, transform):
        super().__init__(".", transform)
        self.G1 = nx.Graph()
        self.G1.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])

        self.G2 = nx.Graph()
        self.G2.add_edges_from([(0, 1), (0, 2)])

        graphs = [torch_geometric.utils.from_networkx(self.G1), torch_geometric.utils.from_networkx(self.G2)]
        self.data, self.slices = self.collate(graphs)
