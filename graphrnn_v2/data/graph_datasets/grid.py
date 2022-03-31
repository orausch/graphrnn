import networkx as nx
import torch_geometric
from torch_geometric import data


class GridDataset(data.InMemoryDataset):
    """
    Creates a dataset of grid graphs.
    """

    min_side = 10
    max_side = 20

    def __init__(self, transform):
        super().__init__(".", transform)
        graphs = [
            torch_geometric.utils.from_networkx(nx.grid_2d_graph(i, j))
            for i in range(self.min_side, self.max_side)
            for j in range(self.min_side, self.max_side)
        ]
        self.data, self.slices = self.collate(graphs)


class SmallGridDataset(GridDataset):
    """
    Creates a dataset of small grid graphs.
    """

    min_side = 2
    max_side = 10
