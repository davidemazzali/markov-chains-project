import numpy as np

def estimate_quality(x,y):
    x = np.array(x)
    y = np.array(y)
    return np.sum(x==y)/len(x)