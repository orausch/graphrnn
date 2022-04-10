import networkx as nx
import torch
import torch_geometric

from graphrnn_v2.data import EncodeGraphRNNFeature
from graphrnn_v2.models.graphrnn_s import GraphRNN_S
from graphrnn_v2.data import RNNTransform
from graphrnn_v2.data import SmallGridDataset
from graphrnn_v2.experiments import eval
from graphrnn_v2.experiments.models import graphrnn_s_small
from graphrnn_v2.stats.stats import GraphStats



def test_graphrnn_s_forward():
    M = 3
    max_n_node = 5
    batch_size = 4

    model = GraphRNN_S(
        adjacency_size=M,
        embed_first=True,
        adjacency_embedding_size=8,
        hidden_size=8,
        num_layers=4,
        output_embedding_size=8,
    )

    input_length = [5, 5, 4, 4]
    input_sequences = torch.Tensor(
        [
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )

    assert input_sequences.size(0) == batch_size
    assert max(input_length) == max_n_node

    output_sequences = model(input_sequences, input_length)
    assert output_sequences.size() == (batch_size, max_n_node, M)
    assert (output_sequences <= 1).all() and (output_sequences >= 0).all()

def test_sample():
    test_dataset = SmallGridDataset(transform=RNNTransform(M=15))
    test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]
    sampler_max_num_nodes = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval.load_model(graphrnn_s_small, "./models/graphrnn_s_small_grid_15.pt", M=15, device=device)

    def sample_no_batch(nb_samples):
        return eval.sample(model, nb_samples, sampler_max_num_nodes, device)

    def sample_batch(nb_samples):
        output_sequences, lengths = model.sample(nb_samples, device, sampler_max_num_nodes)
        adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(output_sequences, lengths)
        return [nx.from_numpy_array(adj.numpy()) for adj in adjs]

    print()
    for nb_samples in [2, 4, 8, 16, 32, 64, 128]:
        graphs_batch = sample_batch(nb_samples)
        graphs_no_batch = sample_no_batch(nb_samples)

        d_s_batch = GraphStats.degree(graphs_batch, graphs_batch)
        d_s_no_batch = GraphStats.degree(graphs_no_batch, graphs_no_batch)

        d_diff = GraphStats.degree(graphs_batch, graphs_no_batch)

        d_t_batch = GraphStats.degree(test_graphs, graphs_batch)
        d_t_no_batch = GraphStats.degree(test_graphs, graphs_no_batch)

        print(f"{d_s_batch=:.4f}, {d_s_no_batch=:.4f}, {d_diff=:.4f}, {d_t_batch=:.4f}, {d_t_no_batch=:.4f}")