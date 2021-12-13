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
