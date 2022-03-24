"""
Clean implementation of learning grid graphs using pytorch geometric
"""

import torch
import torch_geometric.utils
import networkx as nx

from torch_geometric import transforms as T, data

## Dataloader and transformations
class EncodeGraphRNNFeature(T.BaseTransform):
    def __init__(self, M):
        self.M = M

    @staticmethod
    def view_lower_bands(adj, M):
        """
        Uses stride tricks to create a view of adj containing the
        M bands below the diagonal of the given square matrix.
        :param adj: dimension N x N
        :param M: number of bands above the diagonal to return
        :result: dimension N x M; the M bands above the diagonal
        """
        N = adj.shape[1]
        adj = adj.reshape(N, N)
        padded_adj = torch.zeros((N + M - 1, N))
        padded_adj[M - 1 :, :] = adj
        return padded_adj.as_strided(
            size=(N - 1, M), stride=(N + 1, N), storage_offset=1
        )

    def __call__(self, data):
        adj = torch_geometric.utils.to_dense_adj(data.edge_index)
        data.sequences = torch.flip(self.view_lower_bands(adj, self.M), dims=[1])
        data.lengths = torch.ones(adj.shape[1], dtype=torch.long) * self.M
        data.lengths[: self.M - 1] = torch.arange(self.M, dtype=torch.long)[1:]
        return data


class BFS(T.BaseTransform):
    """
    Start a breath first search from a random node and reorder the edge list so
    that the node indices correspond to the breadth-first search order.
    """

    def __call__(self, data):
        x = data.x
        edge_index = data.edge_index
        assert data.is_undirected(), "Transform only works for undirected graphs."
        G = torch_geometric.utils.to_networkx(data, to_undirected=data.is_undirected())

        start_node = torch.randint(0, data.num_nodes, (1,)).item()

        # get the breadth-first search order
        bfs_order = [start_node] + [n for _, n in nx.bfs_edges(G, start_node)]
        perm = torch.tensor(bfs_order).argsort()

        return torch_geometric.data.Data(
            x=x, edge_index=perm[edge_index], num_nodes=data.num_nodes
        )


RNNTransform = T.Compose([BFS(), EncodeGraphRNNFeature(M=4)])


class GridDataset(data.InMemoryDataset):
    """
    Creates a dataset of grid graphs.
    """

    def __init__(self, transform):
        super().__init__(".", transform)
        graphs = [
            torch_geometric.utils.from_networkx(nx.grid_2d_graph(i, j))
            for i in range(10, 20)
            for j in range(10, 20)
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
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
        graphs = [torch_geometric.utils.from_networkx(G)]
        self.data, self.slices = self.collate(graphs)


dataloader = torch_geometric.loader.DataLoader(
    GridDataset(transform=RNNTransform), batch_size=32, shuffle=True
)
# dataloader = torch_geometric.loader.DataLoader(DebugDataset(transform=RNNTransform), batch_size=32, shuffle=True)

for batch in dataloader:
    print(batch.num_graphs)
