"""
Run the graphrnn_v2 model on community graphs.
"""
import os
import time
import itertools
import datetime
import pickle

import networkx as nx
from tqdm import tqdm

import wandb

import torch
from torch.nn.utils import rnn as rnnutils
import torch.nn.functional as F

import torch_geometric
from torch_geometric.loader import DataLoader

from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.data import GridDataset, RNNTransform, EncodeGraphRNNFeature, CommunityDataset
from graphrnn_v2.stats.stats import GraphStats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    BASE_SAVE_PATH = os.path.join("/cluster/scratch/rauscho/final_v2_runs/", args.dataset + "_run_" + str(datetime.datetime.now()))
    os.makedirs(BASE_SAVE_PATH)
    wandb.init(project="graphrnn-reproduction", entity="graphnn-reproduction", job_type="final_" + args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "grid":
        M = 40
        grid_dataset = GridDataset(transform=RNNTransform(M=M))
    elif args.dataset == "community2":
        M = 80
        grid_dataset = CommunityDataset(transform=RNNTransform(M=M))
    else:
        assert False
    wandb.config["M"] = M
   
    train_dataset, test_dataset = torch.utils.data.random_split(
        grid_dataset, [int(0.8 * len(grid_dataset)), len(grid_dataset) - int(0.8 * len(grid_dataset))]
    )
    # use a random sampler to match the paper
    sampler_args = dict(num_samples=32*32)
    sampler = torch.utils.data.RandomSampler(train_dataset, **sampler_args, replacement=True)
    loader_args = dict(batch_size=32, num_workers=4)
    train_dataloader = DataLoader(train_dataset, **loader_args, sampler=sampler)
    test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]
    sampler_max_num_nodes = 1000
    model_args = dict(
        adjacency_size=M,
        embed_first=True,
        adjacency_embedding_size=64,
        hidden_size=128,
        num_layers=4,
        output_embedding_size=64)

    wandb.config.update(model_args)
    model = GraphRNN_S(**model_args)

    # model = GraphRNN(
    #     adjacency_size=M,
    #     embedding_size_graph=64,
    #     hidden_size_graph=128,
    #     num_layers_graph=4,
    #     embedding_size_edge=8,
    #     hidden_size_edge=16,
    #     num_layers_edge=4,
    # )

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

            output_sequences = model(x_padded, lengths)

            # grouped_edge_seqs = model.get_grouped_edge_sequences_from_y(y_padded, lengths)
            # output_sequences = model(x_padded, lengths, grouped_edge_seqs)

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

            if epoch % 100 == 0 and batch_idx == 0:
                model_save_path = os.path.join(BASE_SAVE_PATH, f"model_{epoch}.pt")
                graph_save_path = os.path.join(BASE_SAVE_PATH, f"graphs_{epoch}.dat")
                torch.save(model.state_dict(), model_save_path)
                sample_start_time = time.time()
                # sample some graphs and evaluate them
                output_sequences, lengths = model.sample(1024, device, sampler_max_num_nodes)
                adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(output_sequences, lengths)
                graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]
                with open(graph_save_path, "wb") as f:
                    pickle.dump(graphs, f)

                degree_mmd = GraphStats.degree(test_graphs, graphs)
                clustering = GraphStats.clustering(test_graphs, graphs)
                logging_stats["degree_mmd"] = degree_mmd
                logging_stats["clustering"] = clustering
                logging_stats["sample_time"] = time.time() - sample_start_time

            wandb.log(logging_stats)

        scheduler.step()
