from graphrnn_v2.data import RNNTransform
from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import KRegularDataset
from graphrnn_v2.experiments.models import graphrnn, graphrnn_s

if __name__ == "__main__":
    # GraphRNN_s, k = 3
    M = 50
    dataset = KRegularDataset(transform=RNNTransform(M=M), k=3)
    sampler_max_num_nodes = 150
    model = graphrnn_s(M)
    train_experiment(
        f"3_regular_graphrnn_s",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )

    # GraphRNN, k = 3
    M = 50
    dataset = KRegularDataset(transform=RNNTransform(M=M), k=3)
    sampler_max_num_nodes = 150
    model = graphrnn(M)
    train_experiment(
        f"3_regular_graphrnn",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )

    # GraphRNN, no k
    M = 50
    dataset = KRegularDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 150
    model = graphrnn(M)
    train_experiment(
        f"k_regular",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )
