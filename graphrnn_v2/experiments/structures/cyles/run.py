from graphrnn_v2.data import RNNTransform
from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import CycleDataset
from graphrnn_v2.experiments.models import graphrnn, graphrnn_s

if __name__ == "__main__":
    # Small M, GraphRNN
    M = 5
    dataset = CycleDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 250
    model = graphrnn(M)
    train_experiment(
        f"cycles_small_m",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=False,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )

    # Small M,  GraphRNN_s
    M = 5
    dataset = CycleDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 250
    model = graphrnn_s(M)
    train_experiment(
        f"cycles_small_m_graphrnn_s",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=False,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )

    # Large M
    M = 100
    dataset = CycleDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 250
    model = graphrnn(M)
    train_experiment(
        f"cycles_large_m",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=False,
        num_workers=2,
        plot=True,
        max_epochs=1500
    )
