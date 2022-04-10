from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import SmallEgoDataset
from graphrnn_v2.experiments.models import graphrnn_small, graphrnn_s_small
from graphrnn_v2.data import RNNTransform


if __name__ == "__main__":
    M = 15
    dataset = SmallEgoDataset(transform=RNNTransform(M))
    sampler_max_num_nodes = 20

    model = graphrnn_s_small(M)
    train_experiment(
        f"ego_small_graph_rnn_s",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=3000
    )

    model = graphrnn_small(M)
    train_experiment(
        f"ego_small",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=3000
    )
