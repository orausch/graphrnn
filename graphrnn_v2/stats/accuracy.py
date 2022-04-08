import networkx as nx


class Accuracy:
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        ...


class TreeAccuracy(Accuracy):
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        return sum(nx.is_tree(G) for G in graphs) / len(graphs)


class CycleAccuracy(Accuracy):
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        # Assume graphs are connected.
        return sum(
            nx.is_connected(G) and all(deg == 2 for _, deg in G.degree())
            for G in graphs
        ) / len(graphs)


class KRegularAccuracy(Accuracy):
    @staticmethod
    def measure(graphs: list[nx.Graph], k: int = None) -> float:
        def is_regular(G: nx.Graph) -> bool:
            return all(
                deg == G.degree(0) if k is None else deg == k for _, deg in G.degree()
            )

        return sum(is_regular(G) for G in graphs) / len(graphs)
