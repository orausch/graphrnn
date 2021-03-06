import torch
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
        self.adjacency_size = adjacency_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden = None

        if embed_first:
            self.embedding = nn.Sequential(
                nn.Linear(adjacency_size, adjacency_embedding_size),
                nn.ReLU(),
            )
            input_to_rnn_size = adjacency_embedding_size
        else:
            self.embedding = nn.Identity()
            input_to_rnn_size = adjacency_size

        self.rnn = nn.GRU(
            input_size=input_to_rnn_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.adjacency_mlp = nn.Sequential(
            nn.Linear(hidden_size, output_embedding_size),
            nn.ReLU(),
            nn.Linear(output_embedding_size, adjacency_size),
            nn.Sigmoid(),
        )

    def mask_out_bits_after_length(self, sequences, lengths):
        sequences = pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
        sequences = pad_packed_sequence(sequences, batch_first=True)[0]
        return sequences

    def forward(self, input_sequences, input_length, sampling=False):
        """
        @param input_sequences: (batch_size, max_num_nodes, adjacency_size=M)
            For each graph in the batch, the sequence of adjacency vectors (including the first SOS).
        @param input_length: (batch_size,)
            num_nodes for each graph in the batch. Because graph-sequences where padded to max_num_nodes.
        """
        input_sequences = self.embedding(input_sequences)

        # Pack sequences for RNN efficiency.
        input_sequences = pack_padded_sequence(input_sequences, input_length, batch_first=True, enforce_sorted=False)
        if sampling:
            output_sequences, self.hidden = self.rnn(input_sequences, self.hidden)
        else:
            output_sequences, self.hidden = self.rnn(input_sequences)
        # Unpack RNN output.
        output_sequences, output_length = pad_packed_sequence(output_sequences, batch_first=True)

        # MLP to get adjacency vectors.
        output_sequences = self.adjacency_mlp(output_sequences)

        return self.mask_out_bits_after_length(output_sequences, input_length)

    def sample(self, batch_size, device, max_num_nodes):
        """
        Sample a batch of graph sequences.
        @return: Tensor of size (batch_size, max_num_node, self.adjacency_size) in the same device as the model.

        Note: In the original implementation a max_num_node is used as a placeholder for the generated graphs.
        This makes the assumption that the largest generated graph will have max_num_nodes.

        Instead, this implementation makes the assumption that generated graphs are connected.
        This assumption is implicit in the original codebase.

        In any case one of the above assumptions has to be made to know when the sampler is done generating a graph.
        The disconnected graph assumption can be dropped by adding an SOS flag to the model
        rather than an SOS token which can be confused with a disconnected node.
        """
        input_sequence = torch.ones(batch_size, 1, self.adjacency_size, device=device)  # SOS.
        is_not_eos = torch.ones(batch_size, dtype=torch.long)

        sequences = torch.zeros(batch_size, max_num_nodes, self.adjacency_size)
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)
        node_id = 0  # Id of the node to be added to the sequence. Node 0 is not added.
        with torch.no_grad():
            self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            while is_not_eos.any():
                node_id += 1
                if node_id == max_num_nodes:
                    break

                output_sequence_probs = self.forward(input_sequence, torch.ones(batch_size), sampling=True)
                mask = torch.rand_like(output_sequence_probs)
                output_sequence = torch.gt(output_sequence_probs, mask)

                # Identify the EOS sequences and persist them even if model says otherwise.
                is_not_eos *= output_sequence.any(dim=-1).squeeze().cpu()
                seq_lengths += is_not_eos

                sequences[:, node_id - 1] = output_sequence[:, 0]
                input_sequence = output_sequence.float()

        # Clean irrelevant bits and enforce creation of connected graph.
        # Pack to seq_lengths to include empty sequences. Pack does not support empty sequences.
        sequences = self.mask_out_bits_after_length(sequences, seq_lengths + 1)
        sequences = sequences.tril()

        return sequences[:, : seq_lengths.max()], seq_lengths
