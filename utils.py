import numpy as np

def estimate_quality(x,y):
    x = np.array(x)
    y = np.array(y)
    return (1/len(x)) * np.abs(np.sum(x*y))