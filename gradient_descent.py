import numpy as np


def get_normal(size):
    return np.random.normal(size=size)
    
def func1(a, x):
    v1 = np.dot(a.T, x)
    return np.sum(np.exp(v1) + np.exp(-v1))

def grad_func1(a, x):
    v1 = np.dot(a.T, x)
    v2 = np.exp(v1) - np.exp(-v1)   # m x 1
    return np.dot(a, v2)    # n x m, m x 1

def get_init(shape):
    return np.ones(shape=shape)

def norm(x):
    return np.sqrt(np.sum(np.square(x)))

M, N = 10, 5
ETA = 1e-5
P_STAR = 2 * M
alpha, beta = None, None

def grad_descent(a, x, alpha=0.5, beta=0.9):
    grad = grad_func1(a, x)
    if norm(grad) <= ETA:
        return
    



x0 = get_init((5, ))
a = get_normal((N, M))
grad_descent(a, x0)