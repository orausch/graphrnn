import random
import pytest

import torch
import torch_geometric

from graphrnn_v2.data import DebugDataset, GridDataset, RNNTransform

min_M = {
    "DebugDataset": 3,
    "GridDataset": 40,
}


@pytest.mark.parametrize("dataset_cls", [DebugDataset, GridDataset])
@pytest.mark.parametrize("M", [1, 2, 3, 40, 50])
def test_debug(dataset_cls, M):
    dataset = dataset_cls(transform=RNNTransform(M=M))

    data = random.choice(dataset)

    G = torch_geometric.utils.to_networkx(data)
    print(data.x)
    print(data.y)

    shapes_equal = lambda x, y: len(x) == len(y) and all(x[i] == y[i] for i in range(len(x)))
    assert shapes_equal(data.x.size(), [data.num_nodes, M])
    assert shapes_equal(data.y.size(), [data.num_nodes, M])
    # First input row should be all ones (SOS) and last output row should be all zeros (EOS).
    assert torch.all(data.x[0] == 1)
    assert torch.all(data.y[-1] == 0)

    for idx_a in range(1, data.num_nodes):
        for idx_b in range(idx_a - 1, -1, -1):
            # The indices (i,j) referring to the edge idx_a <-> idx_b.
            i = idx_a
            j = idx_a - idx_b - 1
            print(f"{idx_a} -> {idx_b} in data.x at position {(i,j)}")

            if M >= min_M[dataset_cls.__name__]:  # All edges are represented in data.x
                if idx_a - idx_b <= M:  # Edge captured by the matrix.
                    adjacent = idx_b in G.neighbors(idx_a)
                    assert data.x[i, j] == adjacent
                else:  # Edge not captured by the matrix should be safe.
                    print(torch_geometric.utils.to_dense_adj(data.edge_index))
                    assert idx_b not in G.neighbors(idx_a)
            else:  # Not all edges are represented in data.x
                if idx_a - idx_b <= M:  # Edge captured by the matrix.
                    adjacent = idx_b in G.neighbors(idx_a)
                    assert data.x[i, j] == adjacent
                else:  # Edge not captured by the matrix should not be checked.
                    pass
        if idx_a < M:
            # Same meaning for (i,j) as above.
            assert data.relevant_adj_size[idx_a] == idx_a
            i = idx_a
            for j in range(idx_a, M):
                assert data.x[i, j] == 0


def test_grid_runs():
    dataloader = torch_geometric.loader.DataLoader(GridDataset(transform=RNNTransform(40)), batch_size=32, shuffle=True)

    for batch in dataloader:
        print(batch.num_graphs)
