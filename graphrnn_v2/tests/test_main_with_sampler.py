"""
DELETE ME.

Temporaty "test" until we refactor a train method and use it as a subroutine.
"""
import itertools

import networkx as nx
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.utils import rnn as rnnutils
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graphrnn_v2.data import RNNTransform, EncodeGraphRNNFeature
from graphrnn_v2.data import TriangleDebugDataset, DebugDataset
from graphrnn_v2.models import GraphRNN_S


def test_main():
    M = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DebugDataset(transform=RNNTransform(M=M))
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

    model = GraphRNN_S(
        adjacency_size=M,
        embed_first=True,
        adjacency_embedding_size=16,
        hidden_size=32,
        num_layers=2,
        output_embedding_size=16,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 1000], gamma=0.3)

    model.train()
    model = model.to(device)
    for epoch in tqdm(range(3000)):
        for batch_idx, batch in enumerate(itertools.islice(dataloader, 32)):
            batch = batch.to(device)
            optimizer.zero_grad()

            # (1) Transform the batched graphs to a standard mini-batch of dimensions (B, L, M).
            # Where L is max_num_nodes in the batch.
            lengths = batch.length.cpu()  # torch.split needs a tuple of ints.
            lengths_tuple = tuple(lengths.tolist())
            x_padded = rnnutils.pad_sequence(torch.split(batch.x, lengths_tuple), batch_first=True)
            y_padded = rnnutils.pad_sequence(torch.split(batch.y, lengths_tuple), batch_first=True)

            # Sort batch by graph length, needed for the graph-level RNN.
            sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
            x_padded = x_padded[sorted_idx]
            y_padded = y_padded[sorted_idx]

            output_sequences = model(x_padded, sorted_lengths)

            loss = F.binary_cross_entropy(output_sequences, y_padded)
            loss.backward()
            optimizer.step()
        scheduler.step()

    print(loss)

    output_sequences, lengths = model.sample(4, device)
    for graph_sequence, num_nodes in zip(output_sequences, lengths):
        adj = EncodeGraphRNNFeature.inverse(graph_sequence[: num_nodes - 1])
        graph = nx.from_numpy_array(adj.numpy())
        nx.draw_spectral(graph)
        plt.show()


#   weights = torch.ones_like(padded_y)
#   mask = padded_x.eq(torch.ones(M)).any(dim=-1, keepdim=True)
#   F.binary_cross_entropy(output_sequences, padded_y, weight=weights*mask)
