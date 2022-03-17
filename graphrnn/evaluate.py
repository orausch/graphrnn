import logging
import pickle
import argparse
from typing import Iterator
import networkx as nx
import eval.stats

from tqdm import tqdm
from dataclasses import dataclass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pick_connected_component_new(G):
    adj_list = G.adjacency_list()

    for id,adj in enumerate(adj_list):
        id_min = min(adj)
        if id<id_min and id>=1:
            break
    node_list = list(range(id)) # only include node prior than node "id"
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)

    return G


def load_graph_list(fname,is_real=True):

    with open(fname, "rb") as f:
        graph_list = pickle.load(f)

    for i in range(len(graph_list)):
        # remove any self loops
        edges_with_selfloops = nx.selfloop_edges(graph_list[i])
        graph_list[i].remove_edges_from(edges_with_selfloops)

        if is_real:
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])

    return graph_list


@dataclass(frozen=True)
class GraphStats:
    mmd_degree: float
    mmd_clustering: float
    mmd_4orbits: float 


def get_graph_stats(G_test, G_pred) -> GraphStats:

    mmd_degree = eval.stats.degree_stats(G_test, G_pred)
    mmd_clustering = eval.stats.clustering_stats(G_test, G_pred)

    try:
        mmd_4orbits = eval.stats.orbit_stats_all(G_test, G_pred)
    except:
        mmd_4orbits = -1

    return GraphStats(mmd_degree, mmd_clustering, mmd_4orbits)


@dataclass(frozen=True)
class GraphList:
    graphs: list[nx.Graph]
    time: int
    epoch: int


@dataclass(frozen=True)
class EvaluationConfig:
    truth_graphs: GraphList
    pred_graphs: Iterator[GraphList]


def get_evaluator(config: EvaluationConfig) -> Iterator[GraphStats]:

    # <~ fill this closure with side-effects if you want ~>

    def evaluate() -> Iterator[GraphStats]:

        for pred_graph_list in config.pred_graphs:
            _stats = get_graph_stats(config.truth_graphs.graphs, pred_graph_list.graphs)

            print(f'{_stats.mmd_degree = }\n{_stats.mmd_clustering = }\n{_stats.mmd_4orbits = }')

            yield pred_graph_list, _stats

    return evaluate()



def generate_pred_list(*, graph_type: str = 'grid') -> GraphList:

    epoch_step = 100
    epoch_start = 3000
    epoch_end = 3001

    for epoch in range(epoch_start, epoch_end+1, epoch_step):
        for _time in range(1,4):
            fname = f'{graph_type}/{epoch}_{_time}.dat'

            with open(fname, "rb") as f:
                pred_graph_list = pickle.load(f)

            yield GraphList(pred_graph_list, _time, epoch)


class GraphTypeException(Exception):
    pass

def generate_truth_list(*, graph_type: str = 'grid') -> GraphList:

    if graph_type == 'grid':
        return GraphList(load_graph_list(f'graphs/GraphRNN_RNN_grid_4_128_test_0.dat'), 0, 0)
    elif graph_type.startswith('community'):
        return GraphList(load_graph_list(f'graphs/GraphRNN_RNN_community_4_128_test_0.dat'), 0, 0)
    else:
        raise GraphTypeException('graph_type must be either "grid" or "community"')


def evaluate(*, graph_type: str = 'grid', run_id: str = '0') -> None:

    pred_generator = generate_pred_list(graph_type=graph_type)
    truth_graphs = generate_truth_list(graph_type=graph_type)
    config: EvaluationConfig = EvaluationConfig(truth_graphs, pred_generator)

    evaluation_generator = get_evaluator(config)

    logger.info(f'Evaluating {graph_type}')

    with open(f'eval_{graph_type}_{run_id}.csv', 'w+') as outfile:
        outfile.write('epoch,time,mmd_degree,mmd_clustering,mmd_4orbits\n')
        outfile.flush()

        for pred_graphs, evaluation in evaluation_generator:

            logger.info(f'{pred_graphs.epoch = }\t{pred_graphs.time = }')

            outfile.write(f'{pred_graphs.epoch},{pred_graphs.time},{evaluation.mmd_degree},{evaluation.mmd_clustering},{evaluation.mmd_4orbits}\n')
            outfile.flush()
    






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("graph_type", choices=["grid", "community"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument(
        "--batch_ratio",
        type=int,
        default=32,
        help="How many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches",
    )
    parser.add_argument("--max_prev_node", type=int)
    parser.add_argument("--max_num_node", type=int)

    # parameters for the model sizes
    parser.add_argument("--embedding_size_rnn", type=int, default=64)
    parser.add_argument("--embedding_size_rnn_output", type=int, default=8)
    parser.add_argument("--embedding_size_output", type=int, default=64)
    parser.add_argument("--hidden_size_rnn", type=int, default=128)
    parser.add_argument("--hidden_size_rnn_output", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)

    # Learning rate stuff
    parser.add_argument("--lr_rate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    # parser.add_argument("--milestones", type=float, default=0.003)

    parser.add_argument('--run_id', type=str, default='0')

    args = parser.parse_args()

    evaluate(
        graph_type=args.graph_type,
        run_id=args.run_id,
    )