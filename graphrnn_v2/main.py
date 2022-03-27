"""
Run the graphrnn_v2 model on community graphs.
"""

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graphrnn_v2.models import GraphRNN_S
from graphrnn_v2.data import CommunityDataset, RNNTransform

M = 80


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CommunityDataset(transform=RNNTransform(M=M))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
for epoch in tqdm(range(3000)):
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # sort the batch by the length of the sequences
        # TODO
        # FIXME: Batches in PyG are structured differently from batches in standard pytorch.
        # FIXME: We might have to update the forward method in the model.

        x = batch.x.view(batch.num_graphs, -1, M)
        y = batch.y.view(batch.num_graphs, -1, M)

        output_sequences = model(x, batch.length)

        loss = F.binary_cross_entropy(output_sequences, y)
        loss.backward()
        optimizer.step()
    scheduler.step()
