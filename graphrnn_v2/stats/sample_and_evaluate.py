"""
Quick and dirty script to sample and evaluate a trained model.
Doesn't respect the correct train test split.
"""
import argparse
import os
import pickle
import time

import torch
import torch_geometric
from tqdm import tqdm
import networkx as nx
from graphrnn_v2.experiments.models import graphrnn, graphrnn_s

from graphrnn_v2.stats import GraphStats
from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.data import (
    GridDataset,
    CommunityDataset,
    EgoDataset,
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
    parser.add_argument(
        "--start_epoch", type=int, help="the epoch to start from", required=True
    )
    parser.add_argument("--dataset", type=str, help="name of dataset", required=True)
    parser.add_argument(
        "--model",
        type=str,
        help="name of model",
        required=True,
        choices=["graphrnn", "graphrnn_s"],
    )
    args = parser.parse_args()

    if args.dataset == "grid":
        M = 40
    elif args.dataset == "community":
        M = 80
    elif args.dataset == "ego":
        M = 250
    else:
        raise ValueError("Unknown dataset")

    sampler_max_num_nodes = 1000

    with open(os.path.join(args.base_path, "test_dataset.pkl"), "rb") as f:
        test_graphs = pickle.load(f)

    if args.model == "graphrnn":
        model = graphrnn(M)
    else:
        model = graphrnn_s(M)

    for i in tqdm(range(args.start_epoch, 3001, 100)):
        model.load_state_dict(
            torch.load(os.path.join(args.base_path, "model_{}.pt".format(i)))
        )

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        sample_start_time = time.time()
        output_sequences, lengths = [], []

        graphs = eval.sample(model, 1024, 50, device)

        degree = GraphStats.degree(test_graphs, graphs)
        clustering = GraphStats.clustering(test_graphs, graphs)

        print(
            "epoch {}: degree mmd: {}, clustering mmd: {}".format(i, degree, clustering)
        )
