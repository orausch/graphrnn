"""
Training loops
"""
import os

from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from graphrnn import model, data


def test_mlp_epoch(*, epoch, args, rnn, output, sample_time, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size, args.device)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).to(
        args.device
    )  # normalized prediction score
    y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).to(args.device)  # discrete prediction
    x_step = torch.ones(test_batch_size, 1, args.max_prev_node).to(args.device)
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i : i + 1, :] = torch.sigmoid(y_pred_step)
        x_step = model.sample_sigmoid(args=args, y=y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i : i + 1, :] = x_step
        rnn.hidden = rnn.hidden.data.to(args.device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    G_pred_list = []

    for i in range(test_batch_size):
        adj_pred = data.decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = data.get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


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
    *,
    epoch,
    args,
    rnn,
    output,
    dataloader,
    optimizer_rnn,
    optimizer_output,
    scheduler_rnn,
    scheduler_output,
):

    rnn.train()
    output.train()
    loss_sum = 0

    for batch_idx, data in enumerate(dataloader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]    # Batch of graphs of different sizes.
        y_len_max = max(y_len_unsorted)
        # X is padded to max_n_node in the whole dataset,
        # but can  be padded according to minibatch here.
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), device=args.device)

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = x.to(args.device)
        y = y.to(args.device)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        # Clean the padding that has been transformed by reapplying the padding.
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        if not args.step_scheduler_outside_loop:
            scheduler_output.step()
            scheduler_rnn.step()

        loss_sum += loss.item()
        wandb.log(
            dict(
                loss=loss.item(),
                batch=batch_idx,
                epoch=epoch,
                output_lr=scheduler_output.get_last_lr()[0],
                rnn_lr=scheduler_rnn.get_last_lr()[0],
            )
        )
    if args.step_scheduler_outside_loop:
        scheduler_output.step()
        scheduler_rnn.step()


def train(*, args, dataloader, rnn, output):
    # check if load existing model
    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = optim.lr_scheduler.MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = optim.lr_scheduler.MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)
    save_path = f"{args.graph_save_path}/{args.graph_type}"
    for epoch in tqdm(range(1, args.epochs + 1), unit="epoch"):
        train_epoch(
            epoch=epoch,
            args=args,
            rnn=rnn,
            output=output,
            dataloader=dataloader,
            optimizer_rnn=optimizer_rnn,
            optimizer_output=optimizer_output,
            scheduler_rnn=scheduler_rnn,
            scheduler_output=scheduler_output,
        )

        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            names = []
            print("Evaluating...", end="")
            for sample_time in tqdm(range(1, 4)):
                G_pred = []
                for _ in range(0, args.test_total_size, 16):
                    G_pred.extend(
                        test_mlp_epoch(
                            epoch=epoch,
                            args=args,
                            rnn=rnn,
                            output=output,
                            test_batch_size=16,
                            sample_time=sample_time,
                        )
                    )
                # save graphs
                fname = os.path.join(save_path, f"{epoch}_{sample_time}.dat")
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                print(f"Saving {len(G_pred)} graphs to {fname}")
                data.save_graph_list(G_pred, fname)
                names.append(fname)
            print("Uploading...", end="")
            for name in names:
                wandb.save(name, policy="now")
            print("Done.")
