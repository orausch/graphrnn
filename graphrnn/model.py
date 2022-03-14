"""
Models.
"""
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUPlain(nn.Module):
    """RNN model which can be used for the graph-level RNN and the edge-level RNN"""

    def __init__(
        self, *, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None
    ):
        """
        @param input_size: size of the adjacency vector. The constant M in the paper.
        @param embedding_size:
        @param hidden_size:
        @param num_layers:
        @param has_input:
        @param has_output:
        @param output_size:
        """
        super(GRUPlain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size),
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.25)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("sigmoid"))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=device)

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw


class MLPPlain(nn.Module):
    """MLP used to model the adjacency vector S_i when edges are assumed independent."""

    def __init__(self, *, h_size, embedding_size, y_size):
        """
        @param h_size: size of hidden layer in the graph-level RNN. Used as input.
        @param embedding_size: Size of the MLP linear layer.
        @param y_size: size of the adjacency vector. The constant M in the paper.
        """
        super(MLPPlain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, h):
        y = self.deterministic_output(h)
        return y


def sample_sigmoid(*, args, y, sample, thresh=0.5, sample_time=2):
    """
    Do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    """

    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time > 1:
            y_result = torch.rand(y.size(0), y.size(1), y.size(2)).to(args.device)
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = torch.rand(y.size(1), y.size(2)).to(args.device)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = torch.rand(y.size(0), y.size(1), y.size(2)).to(args.device)
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = (torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).to(args.device)
        y_result = torch.gt(y, y_thresh).float()
    return y_result
