import networkx as nx
import numpy as np
from nx_utils import get_x_star, generate_observation_graph, get_node_label
from metropolis import metropolis_algorithm, sample_from_unif
from utils import estimate_quality
from tqdm import tqdm

# CONSTANTS
d = 3
r = 0.1
N = 1000
n_runs = 100
n_iters = 100

G = generate_observation_graph(d, r, N)

print("Node %d belongs to community %s" % (1, get_node_label(G, 1)))

x_star = get_x_star(G)
qualities = []
for run in tqdm(range(n_runs)):
    x = metropolis_algorithm(G, d, r, sample_from_unif, n_iters)
    quality = estimate_quality(x, x_star)
    qualities.append(quality)
    print("Quality of run %d: %.2f" % (run, quality))