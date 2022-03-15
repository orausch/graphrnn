"""
Experiments entrypoint
"""
import argparse

import wandb

from graphrnn import data, model, train


if __name__ == "__main__":
    wandb.init(project="graphrnn-reproduction", entity="graphnn-reproduction")

    parser = argparse.ArgumentParser()
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("graph_type", choices=["grid", "community2", "community4"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument(
        "--batch_ratio",
        type=int,
        default=32,
        help="How many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches",
    )
    parser.add_argument("--max_prev_node", type=int)
    parser.add_argument("--max_num_node", type=int)
    parser.add_argument("--num_workers", type=int, default=4)

    # parameters for the model sizes
    parser.add_argument("--embedding_size_rnn", type=int, default=64)
    parser.add_argument("--embedding_size_rnn_output", type=int, default=8)
    parser.add_argument("--embedding_size_output", type=int, default=64)
    parser.add_argument("--hidden_size_rnn", type=int, default=128)
    parser.add_argument("--hidden_size_rnn_output", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr_rate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument(
        "--step_scheduler_outside_loop",
        action="store_true",
        help="In the paper codebase, the scheduler is stepped inside the train"
        "loop, which results in a much faster scheduler than they mentioned in"
        "the paper.",
    )

    # train time graph generation
    parser.add_argument(
        "--test_total_size",
        type=int,
        default=1000,
        help="Total number of test graphs to generate",
    )
    parser.add_argument("--epochs_test", type=int, default=100)
    parser.add_argument("--epochs_test_start", type=int, default=100)
    parser.add_argument("graph_save_path", type=str, help="Path to save generated graphs")

    args = parser.parse_args()

    # FIXME force set this for now
    # Should be multiplied by the batch_ratio = 32 i.e. the number of batches per epoch.
    args.milestones = [400, 1000]
    wandb.config["milestones"] = [400, 1000]

    dataloaders = data.create_dataloaders(args)

    rnn = model.GRUPlain(
        input_size=args.max_prev_node,
        embedding_size=args.embedding_size_rnn,
        hidden_size=args.hidden_size_rnn,
        num_layers=args.num_layers,
        has_input=True,
        has_output=False,
    ).to(args.device)
    output = model.MLPPlain(
        h_size=args.hidden_size_rnn,
        embedding_size=args.embedding_size_output,
        y_size=args.max_prev_node,
    ).to(args.device)

    train.train(args=args, dataloader=dataloaders["train"], rnn=rnn, output=output)
