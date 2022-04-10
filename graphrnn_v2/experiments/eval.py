import torch
from tqdm import tqdm

from graphrnn_v2.data import EncodeGraphRNNFeature
from graphrnn_v2.models import GraphRNN, GraphRNN_S

import networkx as nx

def load_model(model_fc, model_path, M, device):
    model = model_fc(M)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def sample(model, nb_samples, sampler_max_num_nodes, device):
    output_sequences, lengths = [], []
    for sample in tqdm(range(nb_samples)):
        seqs, lens = model.sample(1, device, sampler_max_num_nodes)
        output_sequences.append(seqs.squeeze(0))
        lengths.append(lens.squeeze(0))
    adjs = EncodeGraphRNNFeature.get_adjacencies_from_sequences(output_sequences, lengths)
    return [nx.from_numpy_array(adj.numpy()) for adj in adjs]