"""
Run the graphrnn_v2 model on community graphs.
"""
import time
import itertools

from tqdm import tqdm

from torch.cuda import nvtx

import wandb

import torch
from torch.nn.utils import rnn as rnnutils
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.data import CommunityDataset, RNNTransform


def pack_stacked_sequence(stack, lengths, sorted_idx=None):
    """
    Create a PackedSequence from a tensor containing stacked sequences
    :param stack: Tensor of shape (sum(lengths), *)
    :param lengths: lengths of the sequences in the stacked tensor. Must be sorted in descending order.
    :return: PackedSequence
    """
    lengths = torch.tensor(lengths)

    # compute where each sequence starts
    seq_starts = torch.zeros((len(lengths),), dtype=torch.long)
    seq_starts[1:] = lengths.cumsum(0)[:-1]

    if sorted_idx is None:
        # sort the lengths
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
        seq_starts = seq_starts[sorted_idx]
    else:
        sorted_lengths = lengths[sorted_idx]
        seq_starts = seq_starts[sorted_idx]

    idx = []
    batch_sizes = []
    for i in range(sorted_lengths[0]):
        seq_starts = seq_starts[sorted_lengths > i]
        idx.append(seq_starts.clone())
        batch_sizes.append(seq_starts.size(0))
        seq_starts += 1

    idx = torch.cat(idx)
    batch_sizes = torch.tensor(batch_sizes, dtype=torch.long)

    # construct the packed sequence
    return rnnutils.PackedSequence(stack[idx], batch_sizes=batch_sizes, sorted_indices=sorted_idx.to(stack.device))


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

            nvtx.range_push("batch")
            start_time = time.time()

            optimizer.zero_grad()

            nvtx.range_push("sorting")
            # sort lengths
            lengths = torch.tensor(batch.length)
            sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

            x, y = batch.x.to(device), batch.y.to(device)

            # pack the sequences
            packed_x = pack_stacked_sequence(x, lengths, sorted_idx=sorted_idx)
            packed_y = pack_stacked_sequence(y, lengths, sorted_idx=sorted_idx)

            nvtx.range_pop()

            nvtx.range_push("model")
            output_sequences = model(packed_x, sorted_lengths)
            nvtx.range_pop()

            padded_y, _ = rnnutils.pad_packed_sequence(packed_y, batch_first=True)

            loss = F.binary_cross_entropy(output_sequences, padded_y)
            nvtx.range_push("backward")
            loss.backward()
            optimizer.step()
            nvtx.range_pop()

            nvtx.range_pop()

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
