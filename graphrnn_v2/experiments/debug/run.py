from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import SmallGridDataset
from graphrnn_v2.experiments.models import graphrnn_s_small
from graphrnn_v2.data import RNNTransform


if __name__ == "__main__":
    M = 15
    dataset = SmallGridDataset(transform=RNNTransform(M))
    sampler_max_num_nodes = 20
    model = graphrnn_s_small(M)
    train_experiment(
        f"debug",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=False,
        num_workers=0,
        plot=True,
        max_epochs=1000,
        mode="disabled"
    )
