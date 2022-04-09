"""
Run the graphrnn_v2 model on community graphs.
"""
import time

import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric
from matplotlib import pyplot as plt
from torch.nn.utils import rnn as rnnutils
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from graphrnn_v2.data import (
    SmallEgoDataset,
    TriangleDebugDataset,
    MixedDebugDataset,
    SmallGridDataset,
    RNNTransform,
    EncodeGraphRNNFeature,
    DebugDataset,
    LineDebugDataset,
)
from graphrnn_v2.models import GraphRNN
from graphrnn_v2.stats.stats import GraphStats


def plot(graphs):
    plt.figure(figsize=(10, 10))
    for i, graph in enumerate(graphs):
        plt.subplot(4, 2, 2*i + 1)
        nx.draw_spectral(graph, node_size=100)
        plt.subplot(4, 2, 2*i + 2)
        nx.draw(graph, node_size=100)
    plt.show()


if __name__ == "__main__":
    wandb.init(
        project="graphrnn-reproduction", entity="graphnn-reproduction", job_type="v2-twin-rnn-test", mode="online"
    )
    # FIXME: Edit params as you wish.
    M = 15  # 20, 3, 5
    Dataset = SmallGridDataset  # SmallGridDataset  # TriangleDebugDataset  # MixedDebugDataset
    sampler_max_num_nodes = 20  # 20, 5, 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(transform=RNNTransform(M=M))
    train_dataset, test_dataset = dataset, dataset
    sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=32 * 32, replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=2, sampler=sampler)
    test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]
    plot(test_graphs[:4])

    model = GraphRNN(
        adjacency_size=M,
        embedding_size_graph=32,
        hidden_size_graph=64,
        num_layers_graph=4,
        embedding_size_edge=8,
        hidden_size_edge=16,
        num_layers_edge=4,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 1000], gamma=0.3)

    model.train()
    model = model.to(device)
    for epoch in tqdm(range(3000)):
        for batch_idx, batch in enumerate(train_dataloader):

            batch = batch.to(device)
            start_time = time.time()
            optimizer.zero_grad()

            # (1) Transform the batched graphs to a standard mini-batch of dimensions (B, L, M).
            # Where L is max_num_nodes in the batch.
            lengths = batch.length.cpu()  # torch.split needs a tuple of ints.
            lengths_tuple = tuple(lengths.tolist())
            x_padded = rnnutils.pad_sequence(torch.split(batch.x, lengths_tuple), batch_first=True)
            y_padded = rnnutils.pad_sequence(torch.split(batch.y, lengths_tuple), batch_first=True)
            grouped_edge_seqs = model.get_grouped_edge_sequences_from_y(y_padded, lengths)

            output_sequences = model(x_padded, lengths, grouped_edge_seqs)

            loss = F.binary_cross_entropy(output_sequences, y_padded)
            loss.backward()
            optimizer.step()

            batch_time = time.time() - start_time

            logging_stats = dict(
                loss=loss.item(),
                batch=batch_idx,
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                batch_time=batch_time,
                batch_size=batch.num_graphs,
            )

            if epoch % 100 == 0:
                # Compute the epoch NLL. Can be refactored.
                if batch_idx == 0:
                    epoch_nll = 0
                with torch.no_grad():
                    # Only leave relevant bits. (In rows  i < M, remove bits before i).
                    output_sequences *= output_sequences.tril()
                    epoch_nll += (
                        F.binary_cross_entropy(output_sequences, y_padded, reduction="sum").item() / batch.num_graphs
                    )

                if batch_idx == 0:
                    # sample some graphs and evaluate them
                    sample_start_time = time.time()
                    output_sequences, lengths = [], []
                    for sample in range(64):
                        seqs, lens = model.sample(1, device, sampler_max_num_nodes)
                        output_sequences.append(seqs.squeeze(0))
                        lengths.append(lens.squeeze(0))
                    adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(output_sequences, lengths)
                    graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]
                    plot(graphs[:4])

                    degree_mmd = GraphStats.degree(test_graphs, graphs)
                    logging_stats["degree_mmd"] = degree_mmd
                    logging_stats["sample_time"] = time.time() - sample_start_time

                if batch_idx == len(train_dataloader) - 1:
                    logging_stats["epoch_nll"] = epoch_nll / len(train_dataloader)

            wandb.log(logging_stats)

        scheduler.step()
