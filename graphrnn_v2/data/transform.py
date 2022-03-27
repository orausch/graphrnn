import torch
import torch_geometric.utils
import networkx as nx

from torch_geometric import transforms as T


class EncodeGraphRNNFeature(T.BaseTransform):
    def __init__(self, M):
        self.M = M

    @staticmethod
    def view_lower_bands(adj, M):
        """
        Uses stride tricks to create a view of adj containing the
        M bands above the diagonal of the given square matrix.
        :param adj: dimension N x N
        :param M: number of bands above the diagonal to return
        :result: dimension N x M; the M bands above the diagonal
        """
        N = adj.shape[1]
        adj = adj.reshape(N, N)
        padded_adj = torch.zeros((N + M - 1, N))
        padded_adj[M - 1 :, :] = adj
        return padded_adj.as_strided(size=(N - 1, M), stride=(N + 1, N), storage_offset=1)

    def __call__(self, data):
        adj = torch_geometric.utils.to_dense_adj(data.edge_index)
        N = adj.shape[1]
        data.sequences = torch.flip(self.view_lower_bands(adj, self.M), dims=[1])

        # add SOS (row of ones) and EOS (row of zeros)
        data.sequences = torch.cat([torch.ones(1, self.M), data.sequences, torch.zeros(1, self.M)], dim=0)

        # data.lengths = torch.ones(N, dtype=torch.long) * self.M
        # # after the first row the first M - 1 sequences are not full length
        # data.lengths[1 : self.M] = torch.arange(1, self.M, dtype=torch.long)[: min(self.M - 1, N - 1)]

        data.length = data.sequences[:-1].shape[0]

        data.x = data.sequences[:-1]
        data.y = data.sequences[1:]
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

        return torch_geometric.data.Data(x=x, edge_index=perm[edge_index], num_nodes=data.num_nodes)


class RNNTransform(T.Compose):
    def __init__(self, M):
        super(RNNTransform, self).__init__([BFS(), EncodeGraphRNNFeature(M=M)])