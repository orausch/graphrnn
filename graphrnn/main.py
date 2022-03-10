"""
Experiments entrypoint
"""
import argparse
from graphrnn import data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    dataloaders = data.create_dataloaders(args)
    for batch in dataloaders["train"]:
        print(batch)
