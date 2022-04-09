from graphrnn_v2.data import RNNTransform
from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import TreeDataset
from graphrnn_v2.experiments.models import graphrnn, graphrnn_s

if __name__ == "__main__":
    # GraphRNN_s,
    M = 10
    dataset = TreeDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 20
    model = graphrnn_s(M)
    train_experiment(
        f"trees_graphrnn_s",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )
    # GraphRNN,
    M = 10
    dataset = TreeDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 20
    model = graphrnn(M)
    train_experiment(
        f"trees",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )