import networkx as nx


class Accuracy:
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        ...


class TreeAccuracy(Accuracy):
    @staticmethod
    def measure(graphs: list[nx.Graph]) -> float:
        return sum([1 if nx.is_tree(G) else 0 for G in graphs]) / len(graphs)
