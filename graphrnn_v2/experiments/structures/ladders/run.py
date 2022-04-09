from graphrnn_v2.data import RNNTransform
from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import LadderDataset
from graphrnn_v2.experiments.models import graphrnn, graphrnn_s

if __name__ == "__main__":
    # GraphRNN_s,
    M = 10
    dataset = LadderDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 128
    model = graphrnn_s(M)
    train_experiment(
        f"ladder_graphrnn_s",
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
    dataset = LadderDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 128
    model = graphrnn(M)
    train_experiment(
        f"ladder",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )