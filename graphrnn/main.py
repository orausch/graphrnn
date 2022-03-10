"""
Experiments entrypoint
"""
import argparse
from graphrnn import data, model, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", choices=["cpu"])
    parser.add_argument("graph_type", choices=["grid"])
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

    # parameters for the model sizes
    parser.add_argument("--embedding_size_rnn", type=int, default=64)
    parser.add_argument("--embedding_size_rnn_output", type=int, default=8)
    parser.add_argument("--embedding_size_output", type=int, default=64)
    parser.add_argument("--hidden_size_rnn", type=int, default=128)
    parser.add_argument("--hidden_size_rnn_output", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)

    # Learning rate stuff
    parser.add_argument("--lr_rate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    # parser.add_argument("--milestones", type=float, default=0.003)

    args = parser.parse_args()
    # FIXME force set this for now
    args.milestones = [400, 1000]

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
        h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node
    ).to(args.device)

    train.train(args=args, dataloader=dataloaders["train"], rnn=rnn, output=output)
