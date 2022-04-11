import pathlib
import torch
import torch_geometric
import networkx as nx

from pathlib import Path

from graphrnn_v2.data import RNNTransform
from graphrnn_v2.data import SmallGridDataset
from graphrnn_v2.experiments import eval
from graphrnn_v2.experiments.models import *
from graphrnn_v2.stats.stats import GraphStats

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    M = 10
    sampler_max = 20

    filedir = Path(__file__).parent.resolve()

    rnn_mname = Path.joinpath(filedir, "trees_small", "model_at_1400.pt")
    rnn_s_mname = Path.joinpath(filedir, "trees_graphrnn_s_small", "model_at_1400.pt")

    device = torch.device("cpu")
    rnn_model = eval.load_model(
        graphrnn_small,
        rnn_mname,
        M=M,
        device=device,
    )
    rnn_s_model = eval.load_model(
        graphrnn_s_small,
        rnn_s_mname,
        M=M,
        device=device,
    )

    rnn_graphs = eval.sample(rnn_model, 100, sampler_max, device)
    rnn_s_graphs = eval.sample(rnn_s_model, 100, sampler_max, device)

    def plot_(graphs):
        fig = plt.figure(figsize=(5, 10))
        for i, graph in enumerate(graphs):
            plt.subplot(4, 1, i + 1)
            nx.draw(graph, node_size=100)
        return fig

    rnn_fig = plot_(rnn_graphs[-4:])
    rnn_s_fig = plot_(rnn_s_graphs[-4:])

    trees_pdf = PdfPages("trees.pdf")
    trees_pdf.savefig(rnn_fig)
    trees_pdf.savefig(rnn_s_fig)
    trees_pdf.close()
