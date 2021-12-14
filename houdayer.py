from metropolis import metropolis_step
from nx_utils import *
from tqdm.notebook import tqdm
from utils import estimate_quality, visualize_quality, sample_from_unif


def dfs(G, y, i, cluster):
    cluster.add(i)
    for j in G.neighbors(i):
        if y[j] == -1 and j not in cluster:
            dfs(G, y, j, cluster)


def houdayer_move(G, x1, x2):
    y = x1 * x2
    indices = np.argwhere(y == -1)
    i = indices[np.random.randint(len(indices))][0]
    cluster = set()
    dfs(G, y, i, cluster)
    new_x1 = np.copy(x1)
    new_x2 = np.copy(x2)
    for j in cluster:
        new_x1[j] = x2[j]
        new_x2[j] = x1[j]

    return new_x1, new_x2


def houdayer_algorithm(G, d, r, base_chain, n_iters, x_star, houdayer_period=10, use_tqdm=False):
    x1, _, _ = sample_from_unif(x=None, N=len(G.nodes))  # Compute x1_0
    x2, _, _ = sample_from_unif(x=None, N=len(G.nodes))  # Compute x2_0
    neq = np.any(x1 != x2)
    quality_list = []
    iterable = range(n_iters) if not use_tqdm else tqdm(range(n_iters))

    for iter in iterable:
        if neq:
            if iter % houdayer_period != 0:  # Do metropolis step
                x1 = metropolis_step(G, d, r, base_chain, x1)
                x2 = metropolis_step(G, d, r, base_chain, x2)
                neq = np.any(x1 != x2)
            else:  # Do Houdayer move
                x1, x2 = houdayer_move(G, x1, x2)
        else:
            x1 = metropolis_step(G, d, r, base_chain, x1)

        # Quality (evaluated on x1)
        quality = estimate_quality(x1, x_star)
        quality_list.append(quality)
    visualize_quality(quality_list)

    return x1, quality_list
