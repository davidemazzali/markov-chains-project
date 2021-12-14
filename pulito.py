import numpy as np
import networkx as nx
from tqdm import tqdm

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
                    G.add_edge(i,j)
            else:
                if np.random.uniform(0,1)<=b/n:
                    G.add_edge(i, j)
    return G


def sample_from_flip(x):
    i=np.random.randint(n)
    y=x.copy()
    y[i]=-x[i]
    return y

def ratio(x,y):
    i=np.argwhere(x!=y)[0][0]
    ratio=1
    neighbors=set(G.neighbors(i))
    for j in range(n):
        if j!=i:
            if j in neighbors:
                ratio*=(a/b)**(-x[i]*x[j])
            else:
                ratio*=((1-a/n)/(1-b/n))**(-x[i]*x[j])
    return ratio

def acceptance_prob(x,y):
    return min(1, ratio(x,y))

def quality(x_star, x):
    return np.abs(np.sum(x_star*x))/n

def metropolis():
    x=generate_labels()
    for iter in tqdm(range(iterations)):
        #flip
        y=sample_from_flip(x)
        coin=np.random.uniform(0,1)
        if coin<=acceptance_prob(x,y):
            x=y
    return x

def dfs(y,i, cluster):
    cluster.add(i)
    for j in G.neighbors(i):
        if y[j]==-1 and j not in cluster:
            dfs(y, j, cluster)

def houdayer_move(x1, x2):
    y=x1*x2
    indices=np.argwhere(y==-1)
    i=indices[np.random.randint(len(indices))][0]
    cluster=set()
    dfs(y,i, cluster)
    for j in cluster:
            x1[j]=-x1[j]
            x2[j]=-x2[j]



def houdayer(metropolis_steps):
    x1=generate_labels()
    x2=generate_labels()
    neq = np.any(x1 != x2)

    for iter in tqdm(range(iterations)):
        if neq:
            if iter%metropolis_steps==0:
                #make move
                houdayer_move(x1, x2)
                
            y1=sample_from_flip(x1)
            coin=np.random.uniform(0,1)
            if coin<=acceptance_prob(x1,y1):
                x1=y1

            y2=sample_from_flip(x2)
            coin=np.random.uniform(0,1)
            if coin<=acceptance_prob(x2,y2):
                x2=y2
            
            neq = np.all(x1 == x2)
        else:
            y1=sample_from_flip(x1)
            coin=np.random.uniform(0,1)
            if coin<=acceptance_prob(x1,y1):
                x1=y1
            
    return (x1,x2) if neq else (x1,x1)

iterations=10000

num_run=10

n=500
a=5.9
b=0.1

x_star=generate_labels()
G=generate_graph(x_star)




for run in range(num_run):
    x=metropolis()
    print(f'Final quality: {quality(x_star, x)}')

# for run in range(num_run):
#     x1,x2=houdayer(50)
#     print(f'Final quality: {max(quality(x_star, x1),quality(x_star, x2))}')
