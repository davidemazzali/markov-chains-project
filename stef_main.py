import numpy as np
from tqdm import tqdm

def generate_labels(n):
    x = [1 if np.random.randint(2) else -1 for _ in range(n)]
    return x

def generate_graph(a,b,x):
    assert a > b and (a-b)**2 > 2*(a+b)

    n = len(x)
    G = [set() for _ in range(n)]

    for u in range(n):
        for v in range(u+1, n):
            p = (a if x[u] == x[v] else b) / n
            if np.random.binomial(1, p/n):
                G[u].add(v)
                G[v].add(u)
    return G

"""
def quality(x,y):
    x = np.array(x)
    y = np.array(y)
    return np.sum(x==y)/len(x)
"""
def quality(x,y):
    x = np.array(x)
    y = np.array(y)
    return (1/len(x)) * np.abs(np.sum(x*y))

def flip_base_chain(x,y):
    return 1/n

def acceptance_prob(x,y, base_chain, i):
    return min(1, (base_chain(y,x)/(base_chain(x,y)))*ratio_p_flip(y, x, i))

def ratio_p_flip(y,x,i):
    n=len(x)
    ratio=1
    for j in range(n):
        #no self-loops in G
        if j!=i:
            if j in G[i]:
                if x[i]==x[j] and y[i]==y[j]:
                    ratio *= 1
                if x[i]==x[j] and y[i]!=y[j]:
                    ratio *= b/a
                if x[i]!=x[j] and y[i]==y[j]:
                    ratio *= a/b
                if x[i]!=x[j] and y[i]!=y[j]:
                    ratio *= 1
            else:
                if x[i]==x[j] and y[i]==y[j]:
                    ratio *= 1
                if x[i]==x[j] and y[i]!=y[j]:
                    ratio *= (1-b/n)/(1-a/n)
                if x[i]!=x[j] and y[i]==y[j]:
                    ratio *= (1-a/n)/(1-b/n)
                if x[i]!=x[j] and y[i]!=y[j]:
                    ratio *= 1
    return ratio

def sample_from_unif():
    return [np.random.randint(2) for _ in range(n)]

def sample_from_flip(x):
    #select random element in x
    i=np.random.randint(len(x))
    #flip it
    x[i]=-x[i]
    return x, i

def metropolis(x_0, base_chain, num_steps):
    x = x_0
    for s in tqdm(range(num_steps)):
        y, i = sample_from_flip(x)
        coin = np.random.uniform(0,1)
        if coin <= acceptance_prob(x, y, base_chain, i):
            x = y
    return x

def multiple_runs(num_runs, x_0, base_chain, num_steps):
    for i in range(num_runs):
        x=metropolis(x_0, base_chain, num_steps)
        print(f'\nQuality run {i}: {quality(x, x_star)}.')

n = 1000 # number of vertices in the graph
x_star = generate_labels(n) # ground truth labelling of vertices

# paramters
a = n/2
b = n/10

G = generate_graph(a, b, x_star) # graph generated according to stochastic block model

N = 2**n # number of states in the chain

num_runs=5
num_steps=100000

base_chain = lambda x,y : flip_base_chain(x,y)

multiple_runs(num_runs, sample_from_unif(), base_chain, num_steps)