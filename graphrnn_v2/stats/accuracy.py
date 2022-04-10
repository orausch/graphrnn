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


class LadderAccuracy(Accuracy):
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        def is_ladder(G: nx.Graph) -> bool:
            N = len(G)
            if N % 2 != 0:
                return False

            L = nx.ladder_graph(N // 2)  # construct ladder graph of the same size
            return nx.degree_histogram(G) == nx.degree_histogram(
                L
            ) and nx.weisfeiler_lehman_graph_hash(G) == nx.weisfeiler_lehman_graph_hash(
                L
            )

        return sum(is_ladder(G) for G in graphs) / len(graphs)
