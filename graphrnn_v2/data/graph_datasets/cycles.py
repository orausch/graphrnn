import networkx as nx
from torch_geometric import data
from torch_geometric.utils import from_networkx


class CycleDataset(data.InMemoryDataset):
    """
    Create a dataset of cycles.
    """

    min_order = 3
    max_order = 200

    def __init__(self, transform):
        super().__init__(".", transform)
        graphs = [
            from_networkx(nx.cycle_graph(i))
            for i in range(self.min_order, self.max_order + 1)
        ]

        self.data, self.slices = self.collate(graphs)
