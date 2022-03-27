"""
Run the graphrnn_v2 model on community graphs.
"""

from tqdm import tqdm

import torch
from torch.nn.utils import rnn as rnnutils
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.data import CommunityDataset, RNNTransform

M = 80


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CommunityDataset(transform=RNNTransform(M=M))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

model = GraphRNN_S(
    adjacency_size=M,
    embed_first=True,
    adjacency_embedding_size=64,
    hidden_size=64,
    num_layers=2,
    output_embedding_size=64,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 1000], gamma=0.3)

model.train()
model = model.to(device)
for epoch in tqdm(range(3000)):
    for batch in dataloader:
        optimizer.zero_grad()

        # sort lengths
        lengths = batch.length
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)

        # reorder the batch vector
        ordered_batch = sorted_idx[batch.batch]

        # pad to the same length
        padded_x = rnnutils.pad_sequence(
            [batch.x[ordered_batch == i] for i in range(batch.num_graphs)], batch_first=True
        )
        padded_y = rnnutils.pad_sequence(
            [batch.y[ordered_batch == i] for i in range(batch.num_graphs)], batch_first=True
        )

        padded_x, padded_y = padded_x.to(device), padded_y.to(device)

        output_sequences = model(padded_x, sorted_lengths)

        loss = F.binary_cross_entropy(output_sequences, padded_y)
        loss.backward()
        optimizer.step()
    scheduler.step()
