import torch
from torch.nn.utils.rnn import pack_padded_sequence

from graphrnn_v2.models import GraphRNN


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
