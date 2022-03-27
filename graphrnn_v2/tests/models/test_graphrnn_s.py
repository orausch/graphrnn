import torch

from graphrnn_v2.models.graphrnn_s import GraphRNN_S


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
