from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import SmallGridDataset
from graphrnn_v2.experiments.models import graphrnn_small

if __name__ == "__main__":
    Dataset = SmallGridDataset
    M = 15
    sampler_max_num_nodes = 20
    for k in range(10):
        model = graphrnn_small(M)
        train_experiment(
            f"seed_experiment_seed_{k}",
            model,
            M,
            Dataset,
            sampler_max_num_nodes,
            train_test_split=False,
            num_workers=2,
            plot=True,
        )
