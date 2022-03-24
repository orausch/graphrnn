import torch

from graphrnn_v2.models.graphrnn_s import GraphRNN_S

M = 3
batch_size = 4

model = GraphRNN_S(
    adjacency_size=M, embed_first=True, adjacency_embedding_size=8, hidden_size=8, num_layers=4, output_embedding_size=8
)

input_length = [5, 5, 4, 4]
input_sequences = torch.Tensor(
    [
        [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)

output_sequences = model(input_sequences, input_length)
print(output_sequences.size())
