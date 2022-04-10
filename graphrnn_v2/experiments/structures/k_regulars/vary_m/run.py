from graphrnn_v2.data import RNNTransform
from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import SmallKRegularDataset
from graphrnn_v2.experiments.models import graphrnn_small, graphrnn_s_small

if __name__ == "__main__":
    # GraphRNN_s, k = 3
    M = 32
    dataset = SmallKRegularDataset(transform=RNNTransform(M=M))
    sampler_max_num_nodes = 40

    for M in [4, 32, 8, 16]:
        model = graphrnn_s_small(M)
        train_experiment(
            f"3_regular_vary_m_{M}_graphrnn_s",
            model,
            M,
            dataset,
            sampler_max_num_nodes,
            train_test_split=True,
            num_workers=2,
            plot=True,
            max_epochs=2000
        )

        model = graphrnn_small(M)
        train_experiment(
            f"3_regular_vary_m_{M}",
            model,
            M,
            dataset,
            sampler_max_num_nodes,
            train_test_split=True,
            num_workers=2,
            plot=True,
            max_epochs=2000
        )
