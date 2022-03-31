import pickle
import argparse
import networkx as nx

from graphrnn_v2.stats import GraphStats


class GraphEval:
    @staticmethod
    def load(fname: str) -> list[nx.Graph]:
        with open(fname, "rb") as f:
            graphs = pickle.load(f)

        return graphs


if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="predictions file containing serialized list of graphs")
    parser.add_argument("truth", help="truth file containing serialized list of graphs")
    args = parser.parse_args()

    # Format inputs.
    preds = args.predictions
    tests = args.truth

    preds = GraphEval.load(preds)
    tests = GraphEval.load(tests)

    # Compute statistics
    degree_stats = GraphStats.degree(tests, preds)

    print(degree_stats)
