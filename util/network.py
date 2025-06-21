import networkx as nx
import random
from typing import List
from util.agents import BaseAgent  # assumes all agent classes inherit from BaseAgent

class NetworkBuilder:
    """Builds a graph and wires up neighbor lists on a list of agents."""

    def __init__(self, network_type: str, connectivity: float, random_seed: int):
        self.network_type = network_type
        self.connectivity = connectivity
        self.random_seed = random_seed

    def build(self, agents: List[BaseAgent]) -> nx.Graph:
        n = len(agents)
        random.seed(self.random_seed)

        if self.network_type == "random":
            G = nx.erdos_renyi_graph(n, self.connectivity, seed=self.random_seed)

        elif self.network_type == "small_world":
            k = max(2, int(n * self.connectivity))
            if k % 2 == 1:
                k += 1
            k = min(k, n - 1)
            G = nx.watts_strogatz_graph(n, k, self.connectivity, seed=self.random_seed)

        elif self.network_type == "scale_free":
            m = max(1, int(n * self.connectivity / 2))
            G = nx.barabasi_albert_graph(n, m, seed=self.random_seed)

        elif self.network_type == "complete":
            G = nx.complete_graph(n)

        else:
            raise ValueError(f"Unknown network type: {self.network_type}")

        # clear any existing neighbor lists
        for agent in agents:
            agent.neighbors = []

        # wire up neighbors
        for u, v in G.edges():
            agents[u].add_neighbor(agents[v])
            agents[v].add_neighbor(agents[u])

        return G
