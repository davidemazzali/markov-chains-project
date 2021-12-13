import numpy as np
import networkx as nx

def generate_labels():
    x=np.random.randint(2, size=n)
    x=2*x-1
    return x

def generate_graph(x):
    G=nx.Graph()
    for i in range(n):
        G.add_node(i)
        for j in range(i+1, n):
            if x[i]==x[j]:
                if np.random.uniform(0,1)<=a/n:
                    print(f'i:{i}, j:{j}')
                    G.add_edge(i,j)
            else:
                if np.random.uniform(0,1)<=b/n:
                    print(f'i:{i}, j:{j}')
                    G.add_edge(i, j)
    return G


def sample_from_flip(x):
    i=np.random.randint(n)
    print(i)
    y=x.copy()
    y[i]=-x[i]
    return y

def ratio(x,y):
    i=np.argwhere(x!=y)[0][0]
    ratio=1
    for j in range(n):
        if j!=i:
            if j in list(G.neighbors(i)):
                ratio*=(a/b)**(-x[i]*x[j])
            else:
                ratio*=((1-a/n)/(1-b/n))**(-x[i]*x[j])
    return ratio

def acceptance_prob(x,y):
    return min(1, ratio(x,y))


def metropolis():
    x=generate_labels(n)
    for iter in range(iterations):
        #flip
        y=sample_from_flip(x)
        coin=np.random.uniform(0,1)
        if coin<=acceptance_prob(x,y):
            x=y
    return x
            



iterations=10

n=10
a=5.9
b=0.1

x=generate_labels(5)
G=generate_graph(x)
