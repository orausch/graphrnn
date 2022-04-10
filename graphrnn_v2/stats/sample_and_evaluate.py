"""
Quick and dirty script to sample and evaluate a trained model.
Doesn't respect the correct train test split.
"""
import argparse
import os
import time

import torch
import torch_geometric
from tqdm import tqdm
import networkx as nx

from graphrnn_v2.stats import GraphStats
from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.data import (
    GridDataset,
    CommunityDataset,
    RNNTransform,
    EncodeGraphRNNFeature,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        help="base path to model. Models will be loaded from model_{epoch}.pt",
        required=True,
    )
    parser.add_argument("--start_epoch", type=int, help="the epoch to start from", required=True)
    parser.add_argument("--dataset", type=str, help="name of dataset", required=True)

    args = parser.parse_args()

    sampler_max_num_nodes = 1000

    if args.dataset == "grid":
        M = 40
        grid_dataset = GridDataset(transform=RNNTransform(M=M))
    elif args.dataset == "community2":
        M = 80
        grid_dataset = CommunityDataset(transform=RNNTransform(M=M))
    else:
        assert False, "unknown dataset"

    train_dataset, test_dataset = torch.utils.data.random_split(
        grid_dataset,
        [
            int(0.8 * len(grid_dataset)),
            len(grid_dataset) - int(0.8 * len(grid_dataset)),
        ],
    )

    test_graphs = [
        torch_geometric.utils.to_networkx(graph, to_undirected=True)
        for graph in test_dataset
    ]

    model = GraphRNN_S(
        adjacency_size=M,
        embed_first=True,
        adjacency_embedding_size=64,
        hidden_size=128,
        num_layers=4,
        output_embedding_size=64,
    )

    for i in tqdm(range(args.start_epoch, 3001, 100)):
        model.load_state_dict(
            torch.load(
                os.path.join(args.base_path, "model_{}.pt".format(i))
            )
        )

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        sample_start_time = time.time()
        output_sequences, lengths = [], []

        for sample in range(1000):
            seqs, lens = model.sample(1, device, sampler_max_num_nodes)
            output_sequences.append(seqs.squeeze(0))
            lengths.append(lens.squeeze(0))

        adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(
            output_sequences, lengths
        )
        graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]

        degree_mmd = GraphStats.degree(test_graphs, graphs)
        clustering_mmd = GraphStats.clustering(test_graphs, graphs)

        print(
            "epoch {}: degree mmd: {}, clustering mmd: {}".format(
                i, degree_mmd, clustering_mmd
            )
        )
