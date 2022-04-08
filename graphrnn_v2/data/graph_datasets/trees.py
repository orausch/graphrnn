import networkx as nx
from torch_geometric import data
from torch_geometric.utils import from_networkx


class TreeDataset(data.InMemoryDataset):
    """
    Create a dataset of trees.
    """

    min_order = 2
    max_order = 10

    def __init__(self, transform):
        super().__init__(".", transform)
        graphs = []
        for order in range(self.min_order, self.max_order + 1):
            graphs.extend(nx.nonisomorphic_trees(order))

        graphs = [from_networkx(G) for G in graphs]
        self.data, self.slices = self.collate(graphs)
