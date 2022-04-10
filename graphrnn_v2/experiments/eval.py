import networkx as nx
import torch
import torch.nn.functional as F
from torch.nn.utils import rnn as rnnutils
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graphrnn_v2.data import EncodeGraphRNNFeature
from graphrnn_v2.models import GraphRNN


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


def compute_test_nll(model, test_dataset, device, num_workers=0):
    test_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=32 * 32, replacement=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=num_workers, sampler=test_sampler)

    with torch.no_grad():
        test_nll = 0
        for batch_idx, batch in enumerate(test_dataloader):
            batch = batch.to(device)

            lengths = batch.length.cpu()  # torch.split needs a tuple of ints.
            lengths_tuple = tuple(lengths.tolist())
            x_padded = rnnutils.pad_sequence(torch.split(batch.x, lengths_tuple), batch_first=True)
            y_padded = rnnutils.pad_sequence(torch.split(batch.y, lengths_tuple), batch_first=True)

            if isinstance(model, GraphRNN):
                grouped_edge_seqs = model.get_grouped_edge_sequences_from_y(y_padded, lengths)
                output_sequences = model(x_padded, lengths, grouped_edge_seqs)
            else:
                output_sequences = model(x_padded, lengths)

            test_nll += (
                    F.binary_cross_entropy(output_sequences.tril(), y_padded, reduction="sum").item()
                    / batch.num_graphs
            )
        return test_nll / len(test_dataloader)
