import pathlib
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

from graphrnn_v2.data import RNNTransform, EncodeGraphRNNFeature
from graphrnn_v2.models import GraphRNN
from graphrnn_v2.stats.stats import GraphStats


def plot_(graphs):
    plt.figure(figsize=(10, 10))
    for i, graph in enumerate(graphs):
        plt.subplot(2, 2, i + 1)
        nx.draw_spectral(graph, node_size=100)
    plt.show()


def train_experiment(
    name,
    model,
    M,
    Dataset,
    sampler_max_num_nodes,
    train_test_split=True,
    mode="online",
    epoch_checkpoint=100,
    max_epochs=3000,
    num_workers=0,
    save_path=None,
    plot=False,
):
    run = wandb.init(
        project="graphrnn-reproduction",
        entity="graphnn-reproduction",
        job_type="Experiment",
        name=name,
        mode=mode,
        reinit=True,
    )
    with run:
        if save_path is None:
            save_path = pathlib.Path.cwd()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = Dataset(transform=RNNTransform(M=M))
        if train_test_split:
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
            )
        else:
            train_dataset, test_dataset = dataset, dataset

        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=32 * 32, replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=num_workers, sampler=sampler)
        test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]
        if plot:
            plot_(test_graphs[:4])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 1000], gamma=0.3)

        model.train()
        model = model.to(device)
        for epoch in tqdm(range(max_epochs)):
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

                if isinstance(model, GraphRNN):
                    grouped_edge_seqs = model.get_grouped_edge_sequences_from_y(y_padded, lengths)
                    output_sequences = model(x_padded, lengths, grouped_edge_seqs)
                else:
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
                    batch_size=batch.num_graphs,
                )

                if epoch % epoch_checkpoint == 0:
                    if batch_idx == 0:
                        epoch_nll = 0
                    with torch.no_grad():
                        # Only leave relevant bits. (In rows  i < M, remove bits before i).
                        epoch_nll += (
                            F.binary_cross_entropy(output_sequences.tril(), y_padded, reduction="sum").item()
                            / batch.num_graphs
                        )

                    if batch_idx == 0:
                        sample_start_time = time.time()
                        output_sequences, lengths = model.sample(64, device, sampler_max_num_nodes)
                        adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(output_sequences, lengths)
                        graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]
                        if plot:
                            plot_(graphs[:4])
                        degree_mmd = GraphStats.degree(test_graphs, graphs)
                        logging_stats["degree_mmd"] = degree_mmd
                        logging_stats["sample_time"] = time.time() - sample_start_time

                    if batch_idx == 0:
                        path = save_path / name
                        path.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), path / f"model_at_{epoch}.pt")

                    if batch_idx == len(train_dataloader) - 1:
                        logging_stats["epoch_nll"] = epoch_nll / len(train_dataloader)

                wandb.log(logging_stats)

            scheduler.step()
