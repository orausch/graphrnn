"""
Prepare dataset
"""
import pickle
import random

import networkx
import networkx as nx
import numpy as np
import wandb

import scipy.sparse as sp
import torch
import torch.utils.data
from matplotlib import pyplot as plt


def Graph_load(dataset):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    def parse_index_file(filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pickle.load(open("datasets/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datasets/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


def create_graphs(args):
    graphs = []
    if args.graph_type == "grid":
        # Is drawn as a grid with nx.draw_spectral(g).
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 40
        wandb.config["max_prev_node"] = 40

    if args.graph_type == "ego-small":
        """200 ego graphs with 4 ≤ |V| ≤ 18 (Ego-small)."""
        _, _, G = Graph_load(dataset='citeseer')
        G = nx.subgraph(G, max(nx.connected_components(G), key=len))
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        # FIXME: The authors shuffle the dataset before taking a subset this inherently makes reproducibility hard.
        random.shuffle(graphs)

        graphs = graphs[0:200]
        args.max_prev_node = 15
        wandb.config["max_prev_node"] = 15

    elif args.graph_type.startswith("community"):
        if args.graph_type == "community-small":
            num_communities = 2
            p_intra = 0.3
            p_inter = 0.05
            for num_nodes_per_community in [6, 7, 8, 9, 10]:
                for _ in range(100):
                    graphs.append(n_community([num_nodes_per_community]*num_communities, p_intra=p_intra, p_inter=p_inter))
            args.max_prev_node = 20
            wandb.config["max_prev_node"] = 20
            # FIXME: temporary fix. Shuffling should not be dome here.
            random.shuffle(graphs)
        else:
            if args.graph_type == "community-2":
                num_cummunities = 2
                p_intra = 0.3
                p_inter = 0.05
                for num_nodes_per_community in [30,40, 50, 60, 80]:
                    for _ in range(100):
                        graphs.append(n_community([num_nodes_per_community]*num_cummunities,
                                                  p_intra=p_intra, p_inter=p_inter))
            elif args.graph_type == "community-4":
                num_communities = 4
                p_intra = 0.7
                p_inter = 0.01
                c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
                for k in range(3000):
                    graphs.append(n_community(c_sizes, p_intra=p_intra, p_inter=p_inter))
            args.max_prev_node = 100
            wandb.config["max_prev_node"] = 100

    elif args.graph_type == "debug":
        # The graph in figure 1 of the paper and a subset of it to show padding.
        args.max_prev_node = 3
        wandb.config["max_prev_node"] = 3
        g1 = networkx.from_numpy_array(np.array(
            [[0, 1, 1, 0, 0],
             [1, 0, 0, 1, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 1],
             [0, 0, 1, 1, 0]]))
        g2 = networkx.from_numpy_array(np.array(
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 0],
             [0, 1, 0, 0],]))
        for _ in range(args.batch_size*args.batch_ratio):
            graphs.append(g1.copy())
            graphs.append(g2.copy())

    return graphs

def n_community(c_sizes, p_intra, p_inter):
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
    # FIXME: can simply do sum(dict(nx.bfs_successors(G, start_id)).items(), [start_id]).
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbors = dictionary.get(current)
            if neighbors is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbors
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
        non_zero = np.nonzero(adj_slice)[0]         # !The important line here!
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
    adj = adj[1:n, 0 : n - 1]       # can infer connections to last node from prev connections.

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1       # If we take a fixed size (start+max_prev_node) it would be the same anyway.
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
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))      # why matrix vs array?
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
        len_batch = adj_copy.shape[0]   # Here a batch refers to a batch of adjecency vectors of the same graph.
        x_idx = np.random.permutation(adj_copy.shape[0])
        # FIXME: Is permuting the graph then doing a dfs with a random start equivalent to just doing a dfs with a random start?
        # FIXME: The only thing that seems to change is node ordering during the BFS.
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
        # adj_encoded.shape[0] is num_nodes-1 (as the first row has been removed, and is replaced by SOS)
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
            G = nx.from_numpy_matrix(adj_copy_matrix)   # FIXME: Useless matrix vs array casts.
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
    # FIXME: graphs is never shuffled! For grids, this results in the larger ones being exclusively in the tests.
    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    wandb.config["max_num_node"] = args.max_num_node
    splits = train_test_split(graphs)

    dataloaders = {}

    for data, graphs in splits.items():
        dataset = GraphSequenceSampler(graphs, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node)
        # FIXME: can this just be a RandomSampler? Seems true to me.
        sample_strategy = torch.utils.data.WeightedRandomSampler(
            [1.0 / len(dataset) for _ in range(len(dataset))],
            num_samples=args.batch_size * args.batch_ratio,
            replacement=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sample_strategy, num_workers=args.num_workers
        )
        dataloaders[data] = dataloader
        dataloaders[data + '_len'] = len(dataset)

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
