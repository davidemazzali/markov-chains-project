import numpy as np

def generate_labels(n):
    x = [1 if np.random.randint(2) else -1 for _ in range(n)]
    return x

def generate_graph(a,b,x):
    assert a > b

    n = len(x)
    G = [set() for _ in range(n)]

    for u in range(n):
        for v in range(u+1, n):
            p = (a if x[u] == x[v] else b) / n
            if np.random.binomial(1, p/n):
                G[u].add(v)
                G[v].add(u)

def quality(x,y):
    x = np.array(x)
    y = np.array(y)
    return (1/len(x)) * np.abs(np.sum(x*y))

n = 100
x = generate_labels(n)
G = generate_graph(n/2, n/8, x)