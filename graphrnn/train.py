"""
Training loops
"""

from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.nn.functional as F
from torch import optim

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def binary_cross_entropy_weight(y_pred, y, has_weight=False, weight_length=1, weight_max=10):
    """
    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    """
    if has_weight:
        weight = torch.ones(y.size(0), y.size(1), y.size(2))
        weight_linear = torch.arange(1, weight_length + 1) / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(y.size(0), 1, y.size(2))
        weight[:, -1 * weight_length :, :] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.cuda())
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def train_epoch(
    *, epoch, args, rnn, output, dataloader, optimizer_rnn, optimizer_output, scheduler_rnn, scheduler_output
):

    rnn.train()
    output.train()
    loss_sum = 0

    for batch_idx, data in enumerate(bar := tqdm(dataloader, unit="batch")):
        bar.set_description(f"Epoch: {epoch}")
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x.to(args.device)
        y.to(args.device)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        loss_sum += loss.item()
        wandb.log(dict(loss=loss.item(), batch=batch_idx, epoch=epoch))


def train(*, args, dataloader, rnn, output):
    # check if load existing model
    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = optim.lr_scheduler.MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = optim.lr_scheduler.MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)
    train_epoch(
        epoch=0,
        args=args,
        rnn=rnn,
        output=output,
        dataloader=dataloader,
        optimizer_rnn=optimizer_rnn,
        optimizer_output=optimizer_output,
        scheduler_rnn=scheduler_rnn,
        scheduler_output=scheduler_output,
    )
