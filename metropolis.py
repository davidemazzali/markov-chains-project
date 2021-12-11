import numpy as np
from nx_utils import *


# GENERAL FUNCTIONS


def compute_h(G, d, r, i, j):
    e_i_j = has_edge(G, i, j)
    if e_i_j:
        h = np.log(1 / r)
    else:
        a, b = compute_a_b(d, r)
        h = (1 - a / len(G.nodes)) / (1 - b / len(G.nodes))
    return h


def ratio_pi(G, d, r, y, x):
    N = len(G.nodes)
    ratio = 1
    for i in range(N - 1):
        for j in range(i + 1, N):
            if x[i] * x[j] != y[i] * y[j]:
                h_i_j = compute_h(G, d, r, i, j)
                temp = y[i] * y[j] - x[i] * x[j]
                ratio *= h_i_j * temp
    return ratio


def metropolis_algorithm(G, d, r, base_chain, n_iters):
    x, _, _ = sample_from_unif(x=None, N=len(G.nodes))  # Compute x_0
    for iter in range(n_iters):
        # Get new state
        y, psi_x_y, psi_y_x = base_chain(x, len(G.nodes))

        # Decide whether to move or to stay
        coin = np.random.uniform(0, 1)
        acceptance_prob = min(1, (psi_y_x / psi_x_y)) * ratio_pi(G, d, r, y, x)
        if coin <= acceptance_prob:
            x = y
    return x


# UNIFORM BASE CHAIN FUNCTIONS

def sample_from_unif(x, N):
    y = 2 * np.random.randint(2, size=N) - 1
    return y, 1 / N, 1 / N
