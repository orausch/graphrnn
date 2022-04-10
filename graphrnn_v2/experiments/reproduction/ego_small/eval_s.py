import pickle

import torch
import torch_geometric
from tqdm import tqdm

from graphrnn_v2.experiments import eval
from graphrnn_v2.experiments.models import graphrnn_s_small
from graphrnn_v2.stats.stats import GraphStats

# GrpahRNN_S
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = pickle.load(open('./ego_small_graph_rnn_s/test_dataset.pkl', 'rb'))
test_graphs = [torch_geometric.utils.to_networkx(graph, to_undirected=True) for graph in test_dataset]

for epoch in range(500, 3000, 300):
    model = eval.load_model(graphrnn_s_small, f"ego_small_graph_rnn_s/model_at_{epoch}.pt", M=15, device=device)
    test_nll = eval.compute_test_nll(model, test_dataset, device, num_workers=0)
    graphs = eval.sample(model, 1024, 20, device)
    degree = GraphStats.degree(test_graphs, graphs)
    clustering = GraphStats.clustering(test_graphs, graphs)
    print(f' {epoch=:.0f}')
    print(f'{test_nll=:.4f}, {degree=:.4f} {clustering=:.4f}')
