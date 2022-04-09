import torch
import torch_geometric

from graphrnn_v2.data import RNNTransform
from graphrnn_v2.data import SmallGridDataset
from graphrnn_v2.experiments import eval
from graphrnn_v2.experiments.models import graphrnn_small
from graphrnn_v2.stats.stats import GraphStats


if __name__ == '__main__':
    test_dataset = SmallGridDataset(transform=RNNTransform(M=15))
    test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval.load_model(graphrnn_small, "seed_experiment_seed_0/model_at_2900.pt", M=15, device=device)
    graphs = eval.sample(model, 1024, 50, device)

    degree = GraphStats.degree(test_graphs, graphs)
    clustering = GraphStats.clustering(test_graphs, graphs)
    laplacian = GraphStats.laplacian(test_graphs, graphs)
    print(f'{degree:.4f} {clustering:.4f} {laplacian:.4f}')