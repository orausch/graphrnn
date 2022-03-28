"""
Run the graphrnn_v2 model on community graphs.
"""
import time
import itertools

from tqdm import tqdm

import wandb

import torch
from torch.nn.utils import rnn as rnnutils
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.data import CommunityDataset, RNNTransform


if __name__ == "__main__":
    wandb.init(project="graphrnn-reproduction", entity="graphnn-reproduction", job_type="v2_community")
    M = 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CommunityDataset(transform=RNNTransform(M=M))
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

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
        for batch_idx, batch in enumerate(itertools.islice(dataloader, 32)):

            batch = batch.to(device)
            start_time = time.time()
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

            batch_time = time.time() - start_time

            wandb.log(
                dict(
                    loss=loss.item(),
                    batch=batch_idx,
                    epoch=epoch,
                    lr=scheduler.get_last_lr()[0],
                    batch_time=batch_time,
                )
            )

        scheduler.step()
