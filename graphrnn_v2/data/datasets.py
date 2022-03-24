import torch_geometric
import networkx as nx
from torch_geometric import data


class GridDataset(data.InMemoryDataset):
    """
    Creates a dataset of grid graphs.
    """

    def __init__(self, transform):
        super().__init__(".", transform)
        graphs = [
            torch_geometric.utils.from_networkx(nx.grid_2d_graph(i, j)) for i in range(10, 20) for j in range(10, 20)
        ]
        self.data, self.slices = self.collate(graphs)


class DebugDataset(data.InMemoryDataset):
    """
    Creates a dataset containing only the following debug graph

    (0)---(1)
     |     |
     |     |
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
