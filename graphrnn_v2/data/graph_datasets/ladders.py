import networkx as nx
from torch_geometric import data
from torch_geometric.utils import from_networkx


class LadderDataset(data.InMemoryDataset):
    """
    Create a dataset of ladders.
    """

    min_depth = 1
    max_depth = 32

    def __init__(self, transform):
        super().__init__(".", transform)
        graphs = [
            from_networkx(nx.ladder_graph(i))
            for i in range(self.min_depth, self.max_depth + 1)
        ]

        self.data, self.slices = self.collate(graphs)
