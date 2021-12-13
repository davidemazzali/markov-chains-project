import numpy as np
import copy

def generate_labels(n):
    x = [1 if np.random.randint(2) else -1 for _ in range(n)]
    return x

def generate_graph(a,b,x):
    #assert a > b and (a-b)**2 > 2*(a+b)

    n = len(x)
    G = [set() for _ in range(n)]

    for u in range(n):
        for v in range(u+1, n):
            p = (a if x[u] == x[v] else b) / n
            if np.random.binomial(1, p/n):
                if u < v:
                    G[u].add(v)
                else:
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

def flip_base_chain(x,y): # is the number of state, not the number of vertices in the graph
    return 1/n

def acceptance_prob(x,y, base_chain):
    return min(1, ratio(y,x)*base_chain(y,x)/(base_chain(x,y)))

def ratio(x, y): # computes p(x|G) / p(y|G)

    num_ones = np.sum(x == 1)
    num_minus_ones = n - num_ones

    same_pairs = num_ones*(num_ones-1)/2 + num_minus_ones*(num_minus_ones-1)/2
    opposite_pairs = n*(n-1)/2 - same_pairs

    exponen = 0
    theres_edge_same = 0
    theres_edge_opposite = 0
    for i in range(n):
        for j in G[i]:
            exponen += x[i]*x[j]
            if x[i]*x[j] == 1:
                theres_edge_same += 1
            else:
                theres_edge_opposite += 1

    exponen *= np.log(a/b)
    exponen += np.log((1-a/n)/(1-b/n))*((same_pairs - theres_edge_same) - (opposite_pairs - theres_edge_opposite))
    exponen *= 0.5

    num_ones = np.sum(y == 1)
    num_minus_ones = n - num_ones

    same_pairs = num_ones*(num_ones-1)/2 + num_minus_ones*(num_minus_ones-1)/2
    opposite_pairs = n*(n-1)/2 - same_pairs

    exponen_denom = 0
    theres_edge_same = 0
    theres_edge_opposite = 0
    for i in range(n):
        for j in G[i]:
            exponen_denom += y[i]*y[j]
            if y[i]*y[j] == 1:
                theres_edge_same += 1
            else:
                theres_edge_opposite += 1

    exponen_denom *= np.log(a/b)
    exponen_denom += np.log((1-a/n)/(1-b/n))*((same_pairs - theres_edge_same) - (opposite_pairs - theres_edge_opposite))
    exponen_denom *= 0.5

    return np.exp(exponen-exponen_denom)

"""
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
"""

def sample_from_unif():
    return [2*(np.random.randint(2))-1 for _ in range(n)]

def sample_from_flip():
    return np.random.randint(n)

def metropolis(x_0, base_chain, num_steps):
    x = x_0
    for s in range(num_steps):
        if s % (num_steps/10) == 0:
            print(quality(x, x_star))
        
        #y = sample_from_unif()
        y = copy.deepcopy(x)
        i = sample_from_flip()
        y[i] = -x[i]
        coin = np.random.uniform(0,1)
        #print(x, y, acceptance_prob(x, y, base_chain))
        if coin <= acceptance_prob(x, y, base_chain):
            x = y
        else:
            x = x
    return x

def mixed_metropolis_houdayer(x_1, x_2, base_chain, num_steps):
    for s in range(num_steps):
        if s % (num_steps/10) == 0:
            print(quality(x_1, x_star), quality(x_2, x_star))
        
        if s % 10 == 0:
            x_1, x_2 = houda_houda_houdayer_move(x_1, x_2)
        else:
            y_1 = copy.deepcopy(x_1)
            i = sample_from_flip()
            y_1[i] = -x_1[i]
            coin = np.random.uniform(0,1)
            #print(x, y, acceptance_prob(x, y, base_chain))
            if coin <= acceptance_prob(x_1, y_1, base_chain):
                x_1 = y_1
            else:
                x_1 = x_1
            
            y_2 = copy.deepcopy(x_2)
            i = sample_from_flip()
            y_2[i] = -x_2[i]
            coin = np.random.uniform(0,1)
            #print(x, y, acceptance_prob(x, y, base_chain))
            if coin <= acceptance_prob(x_2, y_2, base_chain):
                x_2 = y_2
            else:
                x_2 = x_2
    return x_1, x_2

def dfs(i, y, visited):
    visited.add(i)
    for j in G[i]:
        if y[j] == -1 and j not in visited:
            dfs(j, y, visited)

def houda_houda_houdayer_move(x_1, x_2):
    x_1_prime = copy.deepcopy(x_1)
    x_2_prime = copy.deepcopy(x_2)
    y = [x_1[i]*x_2[i] for i in range(n)]
    S = []
    for i in range(n):
        if y[i] == -1:
            S.append(i)
    i = S[np.random.randint(len(S))]

    visited = set()
    dfs(i, y, visited)

    for j in visited:
        x_1_prime[j] = -x_1[j]
        x_2_prime[j] = -x_2[j]
    
    return x_1_prime, x_2_prime

n = 100 # number of vertices in the graph
x_star = generate_labels(n) # ground truth labelling of vertices

# paramters
a = n/2
b = 1

G = generate_graph(a, b, x_star) # graph generated according to stochastic block model

N = 2**n # number of states in the chain

#base_chain = lambda x,y : unif_base_chain(x, y, N) # create a lambda function for the base chain passing necessary parameters
base_chain = flip_base_chain

print("aaaaaaaaaaaa", 1/np.sqrt(n))
#x = metropolis(sample_from_unif(), base_chain, 1000000)
x_1, x_2 = mixed_metropolis_houdayer(sample_from_unif(), sample_from_unif(), base_chain, 1000000)
print(quality(x_1, x_star), quality(x_2, x_star))
#print("aaaaaaaaaaaa", 1/np.sqrt(n))
#for _ in range(10):
#    print(quality(sample_from_unif(), x_star))