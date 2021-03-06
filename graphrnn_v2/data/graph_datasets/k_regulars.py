import networkx as nx
from torch_geometric import data
from torch_geometric.utils import from_networkx


class KRegularDataset(data.InMemoryDataset):
    """
    Create a dataset of k-regular graphs.
    """

    min_k = 3
    max_k = 10
    min_order = 16
    max_order = 128

    def __init__(self, transform, *, k=None):
        super().__init__(".", transform)
        graphs = (
            [
                nx.random_regular_graph(k, n)
                for k in range(self.min_k, self.max_k + 1)
                for n in range(self.min_order, self.max_order + 1)
                if (n * k) % 2 == 0
            ]
            if k is None
            else [
                nx.random_regular_graph(k, n)
                for n in range(self.min_order, self.max_order + 1)
                if (n * k) % 2 == 0
            ]
        )

        graphs = [from_networkx(G) for G in graphs if nx.is_connected(G)]
        self.data, self.slices = self.collate(graphs)


class SmallKRegularDataset(data.InMemoryDataset):
    """
    Create a dataset of k-regular graphs.
    """
    min_order = 4
    max_order = 32

    def __init__(self, transform):
        super().__init__(".", transform)
        k = 3

        graphs = [
            nx.random_regular_graph(k, n)
            for n in range(self.min_order, self.max_order + 1)
            if (n * k) % 2 == 0
        ]

        graphs = [from_networkx(G) for G in graphs if nx.is_connected(G)]
        self.data, self.slices = self.collate(graphs)
