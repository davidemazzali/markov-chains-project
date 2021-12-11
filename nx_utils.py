import networkx as nx
from networkx.generators.community import stochastic_block_model
import numpy as np


def compute_a_b(d, r):
    a = 2 * d / (r + 1)
    b = r * a
    return a, b


def generate_observation_graph(d, r, N):
    # Compute a, b
    a, b = compute_a_b(d, r)

    # Compute probability of edges
    prob_same_com = a / N
    prob_diff_com = b / N

    # Generate graph
    size = N // 2
    sizes = [size, size]
    p = [[prob_same_com, prob_diff_com], [prob_diff_com, prob_same_com]]
    G = stochastic_block_model(sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)

    return G


def has_edge(G, i, j):
    return G.has_edge(i, j)


def get_node_label(G, i):
    blocks = G.graph['partition']
    if i in blocks[0]: # If node i is in the first partition, label = -1
        return -1
    else: # Otherwise, label = 1
        return 1 


def get_x_star(G):
    return np.array([get_node_label(G, i) for i in range(len(G.nodes))])

