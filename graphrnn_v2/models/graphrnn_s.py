import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class GraphRNN_S(nn.Module):
    def __init__(
        self,
        *,
        adjacency_size: int,
        embed_first: bool,
        adjacency_embedding_size: int,
        hidden_size: int,
        num_layers: int,
        output_embedding_size: int,
    ):
        """
        @param adjacency_size: Size of an adjacency vector. M in the paper.
        @param embed_first: Whether to transform the adjacency vectors before feeding them to the RNN cell.
        @param adjacency_embedding_size: If embed_first, then
            Size of the embedding of the adjacency vectors before feeding it to the RNN cell.
        @param hidden_size: Size of the hidden vectors of the RNN cell.
        @param num_layers: Number of stacked RNN layers
        @param output_embedding_size: Size of the embedding of the edge_level MLP.
        """
        super().__init__()
        if embed_first:
            self.embedding = nn.Sequential(
                nn.Linear(adjacency_size, adjacency_embedding_size),
                nn.ReLU(),
            )
            input_to_rnn_size = adjacency_embedding_size
        else:
            self.embedding = nn.Identity()
            input_to_rnn_size = adjacency_size

        self.rnn = nn.RNN(
            input_size=input_to_rnn_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden = None

        self.adjacency_mlp = nn.Sequential(
            nn.Linear(hidden_size, output_embedding_size),
            nn.ReLU(),
            nn.Linear(output_embedding_size, adjacency_size),
            nn.Sigmoid(),
        )

    def forward(self, input_sequences, input_length):
        """
        @param input_sequences: (batch_size, max_num_nodes, adjacency_size=M)
            For each graph in the batch, the sequence of adjacency vectors (including the first SOS).
        @param input_length: (batch_size,)
            num_nodes for each graph in the batch. Because graph-sequences where padded to max_num_nodes.
        """

        input_sequences = self.embedding(input_sequences)
        input_sequences = pack_padded_sequence(input_sequences, input_length, batch_first=True)
        output_sequences, hidden = self.rnn(input_sequences)
        output_sequences, output_length = pad_packed_sequence(output_sequences, batch_first=True)
        output_sequences = self.adjacency_mlp(output_sequences)
        self.hidden = hidden
        return output_sequences
