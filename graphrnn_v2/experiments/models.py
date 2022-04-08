from graphrnn_v2.models import GraphRNN, GraphRNN_S


def graphrnn(M):
    return GraphRNN(
        adjacency_size=M,
        embedding_size_graph=64,
        hidden_size_graph=128,
        num_layers_graph=4,
        embedding_size_edge=8,
        hidden_size_edge=16,
        num_layers_edge=4,
    )


def graphrnn_small(M):
    return GraphRNN(
        adjacency_size=M,
        embedding_size_graph=32,
        hidden_size_graph=64,
        num_layers_graph=4,
        embedding_size_edge=8,
        hidden_size_edge=16,
        num_layers_edge=4,
    )


def graphrnn_s_small(M):
    return GraphRNN_S(
        adjacency_size=M,
        embed_first=True,
        adjacency_embedding_size=32,
        hidden_size=64,
        num_layers=4,
        output_embedding_size=64,
    )


def graphrnn_s(M):
    return GraphRNN_S(
        adjacency_size=M,
        embed_first=True,
        adjacency_embedding_size=64,
        hidden_size=128,
        num_layers=4,
        output_embedding_size=32,
    )
