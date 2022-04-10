from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import GridDataset
from graphrnn_v2.experiments.models import graphrnn
from graphrnn_v2.data import RNNTransform


if __name__ == "__main__":
    M = 40
    dataset = GridDataset(transform=RNNTransform(M))
    sampler_max_num_nodes = 400
    model = graphrnn(M)
    train_experiment(
        f"full_grid_skander",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=2,
        plot=True,
        max_epochs=3000
    )
