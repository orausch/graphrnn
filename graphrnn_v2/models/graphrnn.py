import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


class GraphRNN(nn.Module):
    def __init__(
        self,
        *,
        adjacency_size: int,
        embedding_size_graph: int,
        hidden_size_graph: int,
        num_layers_graph: int,
        embedding_size_edge: int,
        hidden_size_edge: int,
        num_layers_edge: int,
    ):
        """
        Graph-Level RNN Params:
        @param adjacency_size: M.
        @param embedding_size_graph: Embedding size of the graph-level RNN during:
            M -> embedding -> GRU_graph and GRU_graph -> embedding -> hidden_edge.
        @param hidden_size_graph: Hidden state size of the graph-level RNN.
        @param num_layers_graph: Nb RNN layers of the graph-level RNN.

        Edge-Level RNN Params:
        @param embedding_size_edge: Embedding size of the edge-level RNN during:
            edge (size=1) -> embedding -> GRU_edge and GRU_edge -> embedding -> edge(size=1).
        @param hidden_size_edge: Hidden state size of the edge-level RNN.
        @param num_layers_edge: Nb RNN layers of the edge-level RNN.
        """
        super().__init__()

        self.adjacency_size = adjacency_size
        self.embedding_size_graph = embedding_size_graph
        self.hidden_size_graph = hidden_size_graph
        self.num_layers_graph = num_layers_graph

        self.embedding_size_edge = embedding_size_edge
        self.hidden_size_edge = hidden_size_edge
        self.num_layers_edge = num_layers_edge

        self.hidden_graph = None
        self.hidden_edge = None

        self.adj_embedding = nn.Sequential(
            nn.Linear(adjacency_size, embedding_size_graph),
            nn.ReLU(),
        )

        self.graph_rnn = nn.GRU(
            input_size=embedding_size_graph,
            hidden_size=hidden_size_graph,
            num_layers=num_layers_graph,
            batch_first=True,
        )

        self.graph_output_to_edge_hidden = nn.Sequential(
            nn.Linear(hidden_size_graph, embedding_size_graph),
            nn.ReLU(),
            nn.Linear(embedding_size_graph, hidden_size_edge),
            # FIXME: RELU HERE? The original paper doesn't do so. What's the tradeoff?
            # nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(1, embedding_size_edge),
            nn.ReLU(),
        )

        self.edge_rnn = nn.GRU(
            input_size=embedding_size_edge,
            hidden_size=hidden_size_edge,
            num_layers=num_layers_edge,
            batch_first=True,
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_size_edge, embedding_size_edge),
            nn.ReLU(),
            nn.Linear(embedding_size_edge, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def get_grouped_edge_sequences_from_y(y_padded, lengths):
        seqs = y_padded[:, :, :-1]
        sos = torch.ones(seqs.size(0), seqs.size(1), 1, device=seqs.device)
        seqs = torch.cat((sos, seqs), dim=-1)
        seqs = pack_padded_sequence(seqs, lengths, batch_first=True, enforce_sorted=False)
        return seqs

    @staticmethod
    def mask_out_bits_after_length(sequences, lengths):
        sequences = pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
        sequences = pad_packed_sequence(sequences, batch_first=True)[0]
        return sequences

    def forward(self, graph_input_sequences, input_length, grouped_edge_sequences, sampling=False):
        """
        @param graph_input_sequences: (batch_size, max_num_nodes, adjacency_size=M)
            For each graph in the batch, the sequence of adjacency vectors of node 0 to node (n-2).
            (Thus including the first SOS and excluding the adjacency vector of the last node).
        @param input_length: (batch_size,)
            num_nodes for each graph in the batch. (graph_input_sequences are padded to max_num_nodes.)
        @param grouped_edge_sequences: Sequences of edges grouped per graph.
            Represented as packed sequences of modified adjacencies.
            Each adjacency represents the edge-sequence of a node.
            It includes an SOS (= [1])  and excludes the last edge of the adjacency.
            Use self.get_grouped_edge_sequences_from_y() to obtain it from the target graph sequences.
        """
        graph_input_sequences = self.adj_embedding(graph_input_sequences)
        graph_input_sequences = pack_padded_sequence(
            graph_input_sequences, input_length, batch_first=True, enforce_sorted=False
        )
        if sampling:
            graph_hidden_sequences, self.hidden_graph = self.graph_rnn(graph_input_sequences, self.hidden_graph)
        else:
            graph_hidden_sequences, self.hidden_graph = self.graph_rnn(graph_input_sequences)
        graph_hidden_sequences, output_length = pad_packed_sequence(graph_hidden_sequences, batch_first=True)

        # Transform hidden graph sequences to initial hidden edge sequences:
        edge_hidden_sequences = self.graph_output_to_edge_hidden(graph_hidden_sequences)
        # The edge hidden states are grouped by graph sequence (i.e. a batch for graph_level_rnn).
        # Shape is (batch_size, max_num_nodes, hidden_size_edge)
        # They need to be separated into individual sequences to form a bigger batch.
        # I.e reshape them to  (hidden_layers, \SUM num_nodes_batch, hidden_size_edge)
        edge_hidden_states = pack_padded_sequence(
            edge_hidden_sequences, output_length, batch_first=True, enforce_sorted=False
        ).data
        extra_hidden_layers = torch.zeros(
            self.num_layers_edge - 1,
            edge_hidden_states.size(0),
            edge_hidden_states.size(1),
            device=edge_hidden_states.device,
        )
        edge_hidden_states = torch.cat((edge_hidden_states.unsqueeze(0), extra_hidden_layers), dim=0)

        # Same for edge sequences, the input to the edge-level RNN.
        # They are grouped by graph (i.e. a batch for graph_level_rnn). Shape is (batch_size, max_num_nodes, M).
        # They should undergo a similar transformation as hidden states and become (\SUM num_nodes_batch, M, 1).
        # This is done by self.get_grouped_edge_sequences_from_y()
        edge_input_sequences = grouped_edge_sequences.data.unsqueeze(-1)
        edge_input_sequences = self.edge_embedding(edge_input_sequences)
        edge_hidden_sequences, self.hidden_edge = self.edge_rnn(edge_input_sequences, edge_hidden_states)
        # Shape = (\SUM num_nodes_batch, M, hidden_size_edge).

        output_edge_sequences = self.edge_decoder(edge_hidden_sequences)
        # Shape = ((\SUM num_nodes_batch, M, 1).

        # Re-transform to edge_sequences:
        packed_graph_sequences = output_edge_sequences.squeeze(-1)
        packed_graph_sequences = PackedSequence(
            packed_graph_sequences,
            grouped_edge_sequences.batch_sizes,
            grouped_edge_sequences.sorted_indices,
            grouped_edge_sequences.unsorted_indices,
        )
        graph_sequences, lengths = pad_packed_sequence(packed_graph_sequences, batch_first=True)
        return graph_sequences

    def forward_edge_rnn(self, edge_input_sequences):
        edge_input_sequences = self.edge_embedding(edge_input_sequences)
        output_sequences, self.hidden_edge = self.edge_rnn(edge_input_sequences, self.hidden_edge)
        return self.edge_decoder(output_sequences)

    def sample(self, batch_size, device, max_num_nodes):
        """
        Sample a batch of graph sequences.
        @return: Tensor of size (batch_size, max_num_node, self.adjacency_size) in the same device as the model.

        Assumes that generated graphs are connected. This assumption is implicit in the original codebase.
        """
        graph_input_sequences = torch.ones(batch_size, 1, self.adjacency_size, device=device)  # Graph-SOS.
        sos_edge_sequences = torch.ones(batch_size, 1, 1, device=device)  # Edge-SOS.
        sos_edge_sequences = pack_padded_sequence(
            sos_edge_sequences, torch.ones(batch_size), batch_first=True, enforce_sorted=False
        )

        is_not_eos = torch.ones(batch_size, dtype=torch.long)
        sequences = torch.zeros(batch_size, max_num_nodes, self.adjacency_size, device=device)
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)
        node_id = 0  # Id of the node to be added to the sequence. Node 0 is not added.
        with torch.no_grad():
            self.hidden_graph = torch.zeros(self.num_layers_graph, batch_size, self.hidden_size_graph, device=device)
            while is_not_eos.any():
                node_id += 1
                if node_id == max_num_nodes:
                    break

                output_sequence = torch.zeros(batch_size, 1, self.adjacency_size, device=device)
                prob_edge_0 = self.forward(
                    graph_input_sequences, torch.ones(batch_size), sos_edge_sequences, sampling=True
                )
                edge = torch.gt(prob_edge_0, torch.rand_like(prob_edge_0)).float()
                output_sequence[:, :, 0] = edge.squeeze(-1)
                for edge_idx in range(1, self.adjacency_size):
                    prob_edge = self.forward_edge_rnn(edge)
                    edge = torch.gt(prob_edge, torch.rand_like(prob_edge)).float()
                    output_sequence[:, :, edge_idx] = edge.squeeze(-1)

                # Identify the EOS sequences and persist them even if model says otherwise.
                is_not_eos *= output_sequence.any(dim=-1).squeeze().cpu()
                seq_lengths += is_not_eos

                sequences[:, node_id - 1] = output_sequence[:, 0]
                graph_input_sequences = output_sequence

        # Clean irrelevant bits and enforce creation of connected graph.
        # Pack to seq_lengths to include empty sequences. Pack does not support empty sequences.
        sequences = self.mask_out_bits_after_length(sequences, seq_lengths + 1)
        sequences = sequences.tril()

        return sequences[:, : seq_lengths.max()], seq_lengths
