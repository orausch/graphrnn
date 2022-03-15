"""
Prepare dataset
"""
import pickle

import networkx as nx
import numpy as np
import wandb

import torch
import torch.utils.data
from matplotlib import pyplot as plt


def create_graphs(args):
    graphs = []
    if args.graph_type == "grid":
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 40
        wandb.config["max_prev_node"] = 40

    elif args.graph_type.startswith("community"):
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        # c_sizes = [15] * num_communities    # Easy version.
        for k in range(3000):
            graphs.append(n_community(c_sizes, p_intra=0.7, p_inter=0.05))
        args.max_prev_node = 30
        wandb.config["max_prev_node"] = 30

    return graphs

def n_community(c_sizes, p_intra=0.3, p_inter=0.05):
    graphs = [nx.gnp_random_graph(c_sizes[i], p_intra, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_components(G))
    for i in range(len(communities)):
        nodes1 = communities[i]
        for j in range(i+1, len(communities)):
            nodes2 = communities[j]
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:  # They require connected networks?
                G.add_edge(list(nodes1)[0], list(nodes2)[0])
    return G


def train_test_split(graphs):
    """
    Split the list of graphs
    """
    test_boundary = int(0.8 * len(graphs))
    val_boundary = int(0.2 * len(graphs))

    # note: train and val overlap
    train = graphs[:test_boundary]
    val = graphs[:val_boundary]
    test = graphs[test_boundary:]
    return dict(train=train, val=val, test=test)


def bfs_seq(G, start_id):
    """
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    """
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output


def encode_adj_flexible(adj):
    """
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    """
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0 : n - 1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end - len(adj_slice) + np.amin(non_zero)

    return adj_output


def encode_adj(adj, max_prev_node=10, is_full=False):
    """
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    """
    if is_full:
        max_prev_node = adj.shape[0] - 1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0 : n - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


def decode_adj(adj_output):
    """
    recover to adj from adj_output
    note: here adj_output have shape (n-1)*m
    """
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i, ::-1][output_start:output_end]  # reverse order
    adj_full = np.zeros((adj_output.shape[0] + 1, adj_output.shape[0] + 1))
    n = adj_full.shape[0]
    adj_full[1:n, 0 : n - 1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


class GraphSequenceSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        # for the ith graph in G...
        # adj_all[i] is the adj matrix of the ith graph
        self.adj_all = []
        # len_all[i] is the number of nodes of the ith graph
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())

        # self.n is the maximum number of nodes
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node

        # max_prev_node is called M in the paper
        if max_prev_node is None:
            print("calculating max previous node, total iteration: {}".format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print("max previous node: {}".format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones (SOS)
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])

        # shuffle the adjacency matrix along the node dimension
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)

        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])

        # array of node ids
        x_idx = np.array(bfs_seq(G, start_idx))

        # order according to BFS iteration order
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]

        # TODO: for each row i, once adj_copy[i, j] is zero, all indices
        # following j should also be zero

        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)

        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0 : adj_encoded.shape[0], :] = adj_encoded
        x_batch[1 : adj_encoded.shape[0] + 1, :] = adj_encoded
        return {"x": x_batch, "y": y_batch, "len": len_batch}

    def calc_max_prev_node(self, iter=20000, topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print("iter {} times".format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1 * topk :]
        return max_prev_node


def create_dataloaders(args):
    graphs = create_graphs(args)
    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    wandb.config["max_num_node"] = args.max_num_node
    splits = train_test_split(graphs)

    dataloaders = {}

    for data, graphs in splits.items():
        dataset = GraphSequenceSampler(graphs, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node)
        # FIXME: can this just be a RandomSampler?
        sample_strategy = torch.utils.data.WeightedRandomSampler(
            [1.0 / len(dataset) for _ in range(len(dataset))],
            num_samples=args.batch_size * args.batch_ratio,
            replacement=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sample_strategy, num_workers=args.num_workers
        )
        dataloaders[data] = dataloader

    return dataloaders


# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def get_graph(adj):
    """
    get a graph from zero-padded adj
    :param adj:
    :return:
    """
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G
