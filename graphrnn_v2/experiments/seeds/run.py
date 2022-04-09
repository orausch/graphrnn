from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import SmallGridDataset
from graphrnn_v2.experiments.models import graphrnn_small
from graphrnn_v2.data import RNNTransform


if __name__ == "__main__":
    M = 15
    dataset = SmallGridDataset(transform=RNNTransform(M))
    sampler_max_num_nodes = 20
    for k in range(5):
        model = graphrnn_small(M)
        train_experiment(
            f"fixed_seed_experiment_seed_{k}",
            model,
            M,
            dataset,
            sampler_max_num_nodes,
            train_test_split=False,
            num_workers=2,
            plot=True,
            max_epochs=1000
        )
