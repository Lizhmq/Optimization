import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import flip

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_normal(size):
    return np.random.normal(size=size)
    
def func1(x):
    x1, x2 = x
    res = np.power(x1, 4) / 4 + np.power(x2, 2) / 2
    res += -x1 * x2 + x1 - x2
    return res

def grad_func1(x):
    x1, x2 = x
    g1 = np.power(x1, 3) - x2 + 1
    g2 = x2 - x1 - 1
    return np.array([g1, g2])

def func2(x):
    x1, x2, x3 = x
    v1 = np.power(3 - x1, 2)
    v2 = 7 * np.power(x2 - x1 ** 2, 2)
    v3 = 9 * np.power(x3 - x1 - x2 ** 2, 2)
    return v1 + v2 + v3

def grad_func2(x):
    x1, x2, x3 = x
    g1 = 2 * x1 - 6 + 14 * (x1 ** 2 - x2) * (2 * x1) - 18 * (x3 - x1 - x2 ** 2)
    g2 = 14 * (x2 - x1 ** 2) + 18 * (x3 - x1 - x2 ** 2) * (-2 * x2)
    g3 = 18 * (x3 - x1 - x2 ** 2)
    return np.array([g1, g2, g3])


def H(x):
    x1, x2 = x
    h11 = 3 * np.power(x1, 2)
    h12 = -1
    h21 = -1
    h22 = 1
    return np.array([[h11, h12], [h21, h22]])

def get_init(v):
    if v == 0:
        return np.array([0, 0])
    else:
        return np.array([1.5, 1])

def norm(x):
    return np.sqrt(np.sum(np.square(x)))


def check_grad(x, func, grad_func):
    x0 = x
    f1 = func(x0)
    grad = grad_func(x0)
    dir = np.random.normal(size=x0.shape)
    t = 1e-6
    x1 = x0 + t * dir
    f2 = func(x1)
    print(f2 - f1)
    print(t * np.dot(grad, dir))

ETA = 1e-5


def backtracking(func, gradf, x, dx, alpha=0.3, beta=0.3):
    t = 1.
    while True:
        f1 = func(x) + alpha * t * np.dot(gradf(x), dx)
        f2 = func(x + t * dx)
        if f2 <= f1:
            break
        t *= beta
    return t

def dfp(x, func, grad_func, Hk=np.identity(2)):
    grad_list, f_list = [], []
    d_list, x_list = [], []
    H_list = []
    while True:
        f = func(x)
        gk = grad_func(x)
        dk = -np.dot(Hk, gk)

        grad_list.append(gk)
        f_list.append(f)
        d_list.append(dk)
        x_list.append(x)

        if len(grad_list) > 1:  # rank-1 correction
            deltax = x_list[-1] - x_list[-2]
            deltag = grad_list[-1] - grad_list[-2]
            mid = np.dot(Hk, deltag)
            Hk = Hk + np.outer(deltax, deltax) / np.dot(deltax, deltag)
            Hk = Hk - np.outer(mid, mid) / np.dot(deltag, mid)
        H_list.append(Hk)
        
        try:
            Tmp = np.linalg.cholesky(Hk)
        except:
            print(f"Step {len(H_list)}: Warning: Hk is not positive definite.")

        g_norm = norm(gk)
        if g_norm <= ETA:
            break
        alpha = backtracking(func, grad_func, x, dk)
        x = x + alpha * dk

    return grad_list, f_list, x_list, H_list



# x1 = get_init(0)
# x2 = get_init(1)
# gl, fl, xl, Hl = dfp(x1, func1, grad_func1)
# print(len(gl))
# print(xl)

# gl, fl, xl, Hl = dfp(x2, func1, grad_func1)
# print(len(gl))
# print(xl)


x = np.zeros(3)
gl, fl, xl, Hl = dfp(x, func2, grad_func2, Hk=np.identity(3))
print(len(gl))
print(xl)