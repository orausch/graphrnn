import random
import pytest

import torch
import torch_geometric

from graphrnn_v2.data import DebugDataset, GridDataset, RNNTransform


@pytest.mark.parametrize("dataset_cls", [DebugDataset, GridDataset])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_debug(dataset_cls, M):
    dataset = dataset_cls(transform=RNNTransform(M=M))

    data = random.choice(dataset)

    G = torch_geometric.utils.to_networkx(data)
    print(data.sequences)

    for i in range(data.num_nodes - 1):
        node_idx_of_row = i + 1
        for j in range(M):
            # the index of the node that is referred to at batch.sequences[i, j]
            node_idx = i - j
            print(i, j, "->", node_idx)

            if i < M and j > i:
                # if we are invalid, the value should be zero
                assert data.sequences[i, j] == 0
            else:
                # if valid, the value should be 1 if the nodes are adjacent
                adjacent = node_idx in G.neighbors(node_idx_of_row)
                if adjacent:
                    assert data.sequences[i, j] == 1
                else:
                    assert data.sequences[i, j] == 0


def test_grid_runs():
    dataloader = torch_geometric.loader.DataLoader(GridDataset(transform=RNNTransform(40)), batch_size=32, shuffle=True)

    for batch in dataloader:
        print(batch.num_graphs)
