import random

import networkx as nx
import torch_geometric
from torch_geometric import data


class CommunityDataset(data.InMemoryDataset):
    """
    Creates a dataset of community graphs.
    """

    @staticmethod
    def generate_community_graph(community_sizes, probability_intra_community=0.7, probability_inter_community=0.01):
        graphs = [
            nx.gnp_random_graph(community_sizes[i], probability_intra_community, seed=i)
            for i in range(len(community_sizes))
        ]
        G = nx.disjoint_union_all(graphs)
        communities = [G.subgraph(c) for c in nx.connected_components(G)]

        for i in range(len(communities)):
            community_a = communities[i]
            for j in range(i + 1, len(communities)):
                community_b = communities[j]
                has_inter_edge = False
                for a in community_a.nodes():
                    for b in community_b.nodes():
                        if random.random() < probability_inter_community:
                            G.add_edge(a, b)
                            has_inter_edge = True

                if not has_inter_edge:
                    a = random.choice(list(community_a.nodes()))
                    b = random.choice(list(community_b.nodes()))
                    G.add_edge(a, b)

        return G

    def __init__(self, transform, num_communities=2):
        super().__init__(".", transform)
        community_sizes = random.choices([12, 13, 14, 15, 16, 17], k=num_communities)

        graphs = [
            torch_geometric.utils.from_networkx(
                self.generate_community_graph(community_sizes, probability_inter_community=0.01)
            )
            for _ in range(3000)
        ]

        self.data, self.slices = self.collate(graphs)
