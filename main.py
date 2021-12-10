import numpy as np

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

def quality(x,y):
    x = np.array(x)
    y = np.array(y)
    return (1/len(x)) * np.abs(np.sum(x*y))

#def whatever_base_chain(i,j, params):
#    return transion probability from i to j in base chain given by params

def unif_base_chain(x,y,N): # is the number of state, not the number of vertices in the graph
    return 1/N

def acceptance_prob(x,y, base_chain):
    return min(1, (base_chain(y,x)*numerator_objective_distrib(y))/(base_chain(x,y)*numerator_objective_distrib(x)))

def numerator_objective_distrib(x):
    res = 1
    for i in range(n):
        for j in range(i+1,n):
            term = 0
            if j in G[i]:
                term = np.log(a/b)
            else:
                term = np.log((1-a/n)/(1-b/n))
            res *= np.exp(0.5*term*x[i]*x[j])
    return res

def sample_from_unif():
    return [2*(np.random.randint(2))-1 for _ in range(n)]

def metropolis(x_0, base_chain, num_steps):
    x = x_0
    for s in range(num_steps):
        y = sample_from_unif()
        coin = np.random.uniform(0,1)
        print(x, y, acceptance_prob(x, y, base_chain))
        if coin <= acceptance_prob(x, y, base_chain):
            x = y
        else:
            x = x
    return x

n = 10 # number of vertices in the graph
x_star = generate_labels(n) # ground truth labelling of vertices

# paramters
a = n/2
b = n/10

G = generate_graph(a, b, x_star) # graph generated according to stochastic block model

N = 2**n # number of states in the chain

base_chain = lambda x,y : unif_base_chain(x, y, N) # create a lambda function for the base chain passing necessary parameters

x = metropolis(sample_from_unif(), base_chain, 10)
print(quality(x, x_star))