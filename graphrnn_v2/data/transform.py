import networkx as nx
import torch
import torch_geometric.utils
from torch_geometric import transforms as T


class EncodeGraphRNNFeature(T.BaseTransform):
    def __init__(self, M):
        self.M = M

    @staticmethod
    def extract_bands(adj, M):
        """
        Uses stride tricks to extract the M bands above the diagonal of the
        given square matrix.

        :param adj: dimension N x N
        :param M: number of bands above the diagonal to return
        :returns: dimension (N - 1) x M; the M bands above the diagonal
        """
        N = adj.shape[1]
        # py-g inserts a batch dimension as the first dimension, this removes it
        adj = adj.reshape(N, N)
        padded_adj = torch.zeros((N + M - 1, N))
        padded_adj[M - 1 :, :] = adj
        return padded_adj.as_strided(size=(N - 1, M), stride=(N + 1, N), storage_offset=1)

    @staticmethod
    def bands_to_matrix(bands):
        """
        Given M bands above the diagonal of a square matrix, return the full matrix.

        :param bands: dimension N x M; the M bands above the diagonal
        :returns: the corresponding matrix of dimension N x N
        """
        M = bands.shape[1]
        N = bands.shape[0] + 1
        padded_adj = torch.zeros((N + M - 1, N))
        view = padded_adj.as_strided(size=(N - 1, M), stride=(N + 1, N), storage_offset=1)
        view[:, :] = bands
        return padded_adj[M - 1 :, :]

    @staticmethod
    def inverse(y):
        """
        Inverse of the __call__ method, given the encoded sequence y

        :param y: encoded sequence, without the SOS and EOS tokens
        :returns: the corresponding adjacency matrix
        """
        bands = torch.flip(y, dims=[1])
        adj = EncodeGraphRNNFeature.bands_to_matrix(bands)
        return adj

    @staticmethod
    def get_adjacencies_from_sequences(graph_sequences, lengths):
        """
        Graph_sequences[i] consists of the adjacency vectors of nodes 1, 2, ..., lengths[i]
        In particular it does not contain a vector for node 0.
        FIXME: Can we vectorize the inverse method to get rid of the for loop?
        """
        return [
            EncodeGraphRNNFeature.inverse(graph_sequence[:seq_length])
            for graph_sequence, seq_length in zip(graph_sequences, lengths)
        ]

    def __call__(self, data):
        adj = torch_geometric.utils.to_dense_adj(data.edge_index)
        N = adj.shape[1]
        sequences = torch.flip(self.extract_bands(adj, self.M), dims=[1])

        # Add SOS (row of ones) and EOS (row of zeros).
        sequences = torch.cat([torch.ones(1, self.M), sequences, torch.zeros(1, self.M)], dim=0)

        # FIXME: Wouldn't data.length <- data.num_nodes be more explicit here? We can even omit the line as use num_nodes.
        data.length = sequences[:-1].shape[0]

        data.x = sequences[:-1]
        data.y = sequences[1:]
        return data


class RandomPermute(T.BaseTransform):
    def __call__(self, data):
        x = data.x
        edge_index = data.edge_index
        perm = torch.randperm(data.num_nodes)
        return torch_geometric.data.Data(x=x, edge_index=perm[edge_index], num_nodes=data.num_nodes)


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

        start_node = 0

        # Get the breadth-first search order.
        bfs_order = [start_node] + [n for _, n in nx.bfs_edges(G, start_node)]
        perm = torch.tensor(bfs_order).argsort()
        # FIXME: Shouldn't x be permuted as well? x <- x[bfs_order] would do the job. (It's always none currently but still ...)
        return torch_geometric.data.Data(x=x, edge_index=perm[edge_index], num_nodes=data.num_nodes)


class RNNTransform(T.Compose):
    def __init__(self, M):
        super(RNNTransform, self).__init__([RandomPermute(), BFS(), EncodeGraphRNNFeature(M=M)])
