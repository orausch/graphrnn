import torch
import torch_geometric

from graphrnn_v2.data import SmallEgoDataset
from graphrnn_v2.data import RNNTransform
from graphrnn_v2.experiments import eval
from graphrnn_v2.experiments.models import graphrnn_s_small, graphrnn_small
from graphrnn_v2.stats.stats import GraphStats

if __name__ == '__main__':
    test_dataset = SmallEgoDataset(transform=RNNTransform(M=15))
    test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = eval.load_model(graphrnn_s_small, "graphrnn_v2/experiments/reproduction/ego_small/ego_small_graph_rnn_s/model_at_2900.pt", M=15, device=device)
    model = eval.load_model(graphrnn_small, "graphrnn_v2/experiments/reproduction/ego_small/ego_small/model_at_500.pt", M=15, device=device)
    graphs = eval.sample(model, 1024, 50, device)

    degree = GraphStats.degree(test_graphs, graphs)
    print(f'degree: {degree:4f}')
    clustering = GraphStats.clustering(test_graphs, graphs)
    print(f'clustering: {clustering:4f}')
    laplacian = GraphStats.laplacian(test_graphs, graphs)
    print(f'laplacian: {laplacian:4f}')
    betweenness_centrality = GraphStats.betweenness_centrality(test_graphs, graphs)
    print(f'betweenness_centrality: {betweenness_centrality:4f}')
    eigenvector_centrality = GraphStats.eigenvector_centrality(test_graphs, graphs)
    print(f'eigenvector_centrality: {eigenvector_centrality:4f}')
    density = GraphStats.density(test_graphs, graphs)
    print(f'density: {density:4f}')
    diameter = GraphStats.diameter(test_graphs, graphs)
    print(f'diameter: {diameter:4f}')
    nb_of_nodes = GraphStats.nb_of_nodes(test_graphs, graphs)
    print(f'nb_of_nodes: {nb_of_nodes:4f}')
    nb_of_edges = GraphStats.nb_of_edges(test_graphs, graphs)
    print(f'nb_of_edges: {nb_of_edges:4f}')
