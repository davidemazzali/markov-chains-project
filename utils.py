import numpy as np
import matplotlib.pyplot as plt


def estimate_quality(x, y):
    x = np.array(x)
    y = np.array(y)
    return (1 / len(x)) * np.abs(np.sum(x * y))


# VISUALIZATION
def visualize_quality(quality_list):
    plt.rcParams["figure.figsize"] = (20, 3)
    plt.ylabel("Quality")
    plt.xlabel("Metropolis step")
    plt.plot(quality_list)
    plt.show()

# BASE CHAINS

def sample_from_unif(x, N=None):
    if N is None:
        N = len(x)
    y = 2 * np.random.randint(2, size=N) - 1
    return y, 1 / N, 1 / N


def sample_from_flip(x, N=None):
    if N is None:
        N = len(x)
    # select random element in x
    i = np.random.randint(len(x))
    # flip it
    #y = np.copy(x)
    #y[i] *= -1
    return i, 1 / N, 1 / N