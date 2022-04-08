import networkx as nx


class Accuracy:
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        ...


class TreeAccuracy(Accuracy):
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        return sum([nx.is_tree(G) for G in graphs]) / len(graphs)


class CycleAccuracy(Accuracy):
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        # Assume graphs are connected.
        assert all(nx.is_connected(G) for G in graphs)
        return sum([all(deg == 2 for _, deg in G.degree()) for G in graphs]) / len(
            graphs
        )
