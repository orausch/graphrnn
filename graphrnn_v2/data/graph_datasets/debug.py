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


class TriangleDebugDataset(data.InMemoryDataset):
    """
    Creates a dataset containing only the following triangle debug graph. Permutations have no effect.

    (0)---(1)
     |   /
     | /
    (2)
    """

    def __init__(self, transform):
        super().__init__(".", transform)
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (0, 2), (1, 2)])
        graphs = [torch_geometric.utils.from_networkx(self.G)]
        self.data, self.slices = self.collate(graphs)


class MixedDebugDataset(data.InMemoryDataset):
    """
    Creates a dataset containing a batch of size 32 with the following two debug graphs (16 of each).
    That is to check how graphs of different sizes aggregate.

    (0)---(1)               (0)---(1)
     |     |                 |
    (2)---(3)               (2)
      \   /
       (4)

    """

    def __init__(self, transform):
        super().__init__(".", transform)
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
        g1 = torch_geometric.utils.from_networkx(g1)

        g2 = nx.Graph()
        g2.add_edges_from([(0, 1), (0, 2)])
        g2 = torch_geometric.utils.from_networkx(g2)

        graphs = [g1 for _ in range(16)] + [g2 for _ in range(16)]
        self.data, self.slices = self.collate(graphs)
