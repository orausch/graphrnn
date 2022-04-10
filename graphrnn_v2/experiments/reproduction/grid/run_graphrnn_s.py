from graphrnn_v2.experiments.train import train_experiment
from graphrnn_v2.data import GridDataset
from graphrnn_v2.experiments.models import graphrnn_s
from graphrnn_v2.data import RNNTransform


if __name__ == "__main__":
    M = 40
    dataset = GridDataset(transform=RNNTransform(M))
    sampler_max_num_nodes = 400
    model = graphrnn_s(M)
    train_experiment(
        f"full_grid_graphrnn_s",
        model,
        M,
        dataset,
        sampler_max_num_nodes,
        train_test_split=True,
        num_workers=4,
        plot=False,
        max_epochs=3000,
        mode="offline"
    )
