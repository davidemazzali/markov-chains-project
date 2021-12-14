import numpy as np
from nx_utils import *
from tqdm.notebook import tqdm
from utils import estimate_quality, visualize_quality, sample_from_unif
import matplotlib.pyplot as plt


# GENERAL FUNCTIONS


def compute_h(G, d, r, i, j):
    e_i_j = has_edge(G, i, j)
    if e_i_j:
        h = np.log(1 / r)
    else:
        a, b = compute_a_b(d, r)
        h = np.log((1 - a / len(G.nodes)) / (1 - b / len(G.nodes)))
    return h


def ratio_pi_flip(G, a, b, i, x):
    N = len(G.nodes)

    ratio = 1
    neighbors = set(G.neighbors(i))
    for j in range(N):
        if j != i:
            if j in neighbors:
                ratio *= (a / b) ** (-x[i] * x[j])
            else:
                ratio *= ((1 - a / N) / (1 - b / N)) ** (-x[i] * x[j])
    return ratio


def ratio_pi(G, d, r, y, x):
    N = len(G.nodes)

    if np.sum(x != y) == 1:
        return ratio_pi_flip(G, d, r, y, x)

    sum = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            if x[i] * x[j] != y[i] * y[j]:
                h_i_j = compute_h(G, d, r, i, j)
                temp = y[i] * y[j] - x[i] * x[j]
                sum += h_i_j * temp

    ratio = np.exp(sum)
    return ratio


def metropolis_step(G, a, b, base_chain, x):
    # Get new state
    i, psi_x_y, psi_y_x = base_chain(x, len(G.nodes))

    # Decide should I stay or should I go
    coin = np.random.uniform(0, 1)
    acceptance_prob = min(1, (psi_y_x / psi_x_y) * ratio_pi_flip(G, a, b, i, x))
    if coin <= acceptance_prob:
        x[i] *= -1

    return x





def metropolis_algorithm(G, d, r, base_chain, n_iters, x_star, use_tqdm=False):
    x, _, _ = sample_from_unif(x=None, N=len(G.nodes))  # Compute x_0
    a, b = compute_a_b(d, r)
    quality_list = []
    iterable = range(n_iters) if not use_tqdm else tqdm(range(n_iters))
    for iter in iterable:
        # New state
        x = metropolis_step(G, a, b, base_chain, x)

        # Quality
        quality = estimate_quality(x, x_star)
        quality_list.append(quality)
    visualize_quality(quality_list)

    return x, quality_list







