import networkx as nx
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence

from graphrnn_v2.models import GraphRNN
from graphrnn_v2.data import EncodeGraphRNNFeature
from graphrnn_v2.models.graphrnn_s import GraphRNN_S
from graphrnn_v2.data import RNNTransform
from graphrnn_v2.data import SmallGridDataset
from graphrnn_v2.experiments import eval
from graphrnn_v2.experiments.models import graphrnn_small
from graphrnn_v2.stats.stats import GraphStats




def test_graphrnn_forward():
    M = 3
    max_n_node = 4
    batch_size = 4

    model = GraphRNN(
        adjacency_size=M,
        embedding_size_graph=16,
        hidden_size_graph=32,
        num_layers_graph=2,
        embedding_size_edge=4,
        hidden_size_edge=8,
        num_layers_edge=2,
    )
    graph_sequences = torch.Tensor(
        [
            [[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        ]
    )
    lengths = [3, 4, 2, 4]
    x = graph_sequences[:, :-1]
    y = graph_sequences[:, 1:]

    graph_input_sequences = x

    edge_input_sequences = model.get_grouped_edge_sequences_from_y(y, lengths)

    assert graph_input_sequences.size(0) == batch_size
    assert max(lengths) == max_n_node

    output_sequences = model(graph_input_sequences, lengths, edge_input_sequences)
    assert output_sequences.size() == (batch_size, max_n_node, M)
    assert (output_sequences <= 1).all() and (output_sequences >= 0).all()


def plot(graphs):
    plt.figure(figsize=(10, 10))
    for i, graph in enumerate(graphs[:4]):
        plt.subplot(4, 2, 2*i + 1)
        nx.draw_spectral(graph, node_size=100)
        plt.subplot(4, 2, 2*i + 2)
        nx.draw(graph, node_size=100)
    plt.show()

def test_sample():
    test_dataset = SmallGridDataset(transform=RNNTransform(M=15))
    test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]
    sampler_max_num_nodes = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval.load_model(graphrnn_small, "./models/graphrnn_small_grid_15.pt", M=15, device=device)

    def sample_no_batch(nb_samples):
        return eval.sample(model, nb_samples, sampler_max_num_nodes, device)

    def sample_batch(nb_samples):
        output_sequences, lengths = model.sample(nb_samples, device, sampler_max_num_nodes)
        adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(output_sequences, lengths)
        return [nx.from_numpy_array(adj.numpy()) for adj in adjs]

    print()
    for nb_samples in [4, 8, 16, 32, 64, 128]:
        graphs_batch = sample_batch(nb_samples)
        graphs_no_batch = sample_no_batch(nb_samples)
        plot(graphs_batch)

        d_s_batch = GraphStats.degree(graphs_batch, graphs_batch)
        d_s_no_batch = GraphStats.degree(graphs_no_batch, graphs_no_batch)

        d_diff = GraphStats.degree(graphs_batch, graphs_no_batch)

        d_t_batch = GraphStats.degree(test_graphs, graphs_batch)
        d_t_no_batch = GraphStats.degree(test_graphs, graphs_no_batch)

        print(f"{d_s_batch=:.8f}, {d_s_no_batch=:.8f}, {d_diff=:.8f}, {d_t_batch=:.8f}, {d_t_no_batch=:.8f}")