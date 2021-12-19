from metropolis import metropolis_step
from nx_utils import *
from tqdm.notebook import tqdm
from utils import estimate_quality, visualize_quality, sample_from_unif
from networkx.algorithms import node_connected_component
import networkx as nx
import time


def dfs(G, y, i, cluster):
    cluster.add(i)
    for j in G.neighbors(i):
        if y[j] == -1 and j not in cluster:
            dfs(G, y, j, cluster)


def houdayer_move_v1(G, x1, x2):
    y = x1 * x2
    indices = np.argwhere(y == -1).flatten()
    i = indices[np.random.randint(len(indices))]
    cluster = set()
    dfs(G, y, i, cluster)
    cluster = list(cluster)
    #print("Houdayer - Flipping %d nodes" % len(cluster))
    x1[cluster] *= -1
    x2[cluster] *= -1

    return x1, x2


def houdayer_move_v2(G, x1, x2):
    y = x1 * x2
    i = np.random.choice(np.argwhere(y == -1).flatten())
    removed_indices = np.argwhere(y == 1).flatten()
    temp_G = nx.Graph.copy(G)
    temp_G.remove_nodes_from(removed_indices)
    cluster = node_connected_component(temp_G, i)
    new_x1 = np.copy(x1)
    new_x2 = np.copy(x2)
    print("Houdayer - Flipping %d nodes" % len(cluster))
    for j in cluster:
        assert x2[j] != x1[j]
        new_x1[j] = x2[j]
        new_x2[j] = x1[j]

    return new_x1, new_x2


def houdayer_algorithm(G, d, r, base_chain, n_iters, x_star, G_np, houdayer_period=10, use_tqdm=False):
    x1, _, _ = sample_from_unif(x=None, N=len(G.nodes))  # Compute x1_0
    x2, _, _ = sample_from_unif(x=None, N=len(G.nodes))  # Compute x2_0
    a, b = compute_a_b(d, r)
    neq = np.any(x1 != x2)
    quality_list = []
    iterable = range(n_iters) if not use_tqdm else tqdm(range(n_iters))

    for iter in iterable:
        if neq:
            if iter % houdayer_period == 0:  # Do metropolis step
                #start = time.time()
                x1, x2 = houdayer_move_v1(G, x1, x2)
                #end = time.time()
                #print("Time for operation: %f" % (end - start))
            
            x1 = metropolis_step(G, a, b, base_chain, x1, G_np)
            x2 = metropolis_step(G, a, b, base_chain, x2, G_np)
            neq = np.any(x1 != x2)
        else:
            x1 = metropolis_step(G, d, r, base_chain, x1, G_np)

        # Quality (evaluated on x1)
        quality = estimate_quality(x1, x_star)
        quality_list.append(quality)
    #visualize_quality(quality_list)

    return x1, quality_list
