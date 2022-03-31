"""
Run the graphrnn_v2 model on community graphs.
"""
import itertools
import time

import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils import rnn as rnnutils
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graphrnn_v2.data import (
    TriangleDebugDataset,
    MixedDebugDataset,
    SmallGridDataset,
    SmallEgoDataset,
    RNNTransform,
    EncodeGraphRNNFeature,
)
from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.stats.stats import GraphStats


def plot(graphs, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    for i, graph in enumerate(graphs):
        plt.subplot(2, 2, i + 1)
        nx.draw_spectral(graph, node_size=100)
    plt.show()


if __name__ == "__main__":
    wandb.init(project="graphrnn-reproduction", entity="graphnn-reproduction", job_type="v2-test")
    # FIXME: Edit params as you wish.
    M = 15  # 20, 3, 5
    Dataset = SmallEgoDataset  # SmallGridDataset  # TriangleDebugDataset  # MixedDebugDataset
    sampler_max_num_nodes = 30  # 20, 5, 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_dataset = Dataset(transform=RNNTransform(M=M))
    train_dataset, test_dataset = torch.utils.data.random_split(
        grid_dataset, [int(0.8 * len(grid_dataset)), len(grid_dataset) - int(0.8 * len(grid_dataset))]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)
    test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]
    plot(test_graphs[:4], "Test graphs")

    model = GraphRNN_S(
        adjacency_size=M,
        embed_first=True,
        adjacency_embedding_size=64,
        hidden_size=128,
        num_layers=4,
        output_embedding_size=64,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 1000], gamma=0.3)

    model.train()
    model = model.to(device)
    for epoch in tqdm(range(3000)):
        for batch_idx, batch in enumerate(itertools.islice(train_dataloader, 32)):

            batch = batch.to(device)
            start_time = time.time()
            optimizer.zero_grad()

            # (1) Transform the batched graphs to a standard mini-batch of dimensions (B, L, M).
            # Where L is max_num_nodes in the batch.
            lengths = batch.length.cpu()  # torch.split needs a tuple of ints.
            lengths_tuple = tuple(lengths.tolist())
            x_padded = rnnutils.pad_sequence(torch.split(batch.x, lengths_tuple), batch_first=True)
            y_padded = rnnutils.pad_sequence(torch.split(batch.y, lengths_tuple), batch_first=True)

            output_sequences = model(x_padded, lengths)

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
            )

            if epoch % 100 == 0 and batch_idx == 0:
                # sample some graphs and evaluate them
                output_sequences, lengths = model.sample(256, device, sampler_max_num_nodes)
                adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(output_sequences, lengths)
                graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]
                plot(graphs[:4], "Sampled graphs")

                degree_mmd = GraphStats.degree(test_graphs, graphs)
                logging_stats["degree_mmd"] = degree_mmd

            wandb.log(logging_stats)

        scheduler.step()
