import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from scipy import optimize

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_normal(size):
    return np.random.normal(size=size)

N = 6

def get_init():
    return -np.ones(N)

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


set_seed(233)
alpha = 0.08
ETA = 1e-5
A = np.random.normal(size=(N, N))
A = (A + A.T) / 2
b = np.random.normal(size=(N,))
lambd = 0.1


def func(x):
    v1 = np.sum(np.power(np.dot(A, x) - b, 2)) / 2
    v2 = lambd * np.sum(np.abs(x))
    return v1 + v2

def grad_func(x):
    return grad1(x) + lambd * np.sign(x)

def grad1(x):
    return np.dot(A.T, np.dot(A, x) - b)

def geng1(x0, alpha=0.02):
    # def f(x):
    #     v1 = func(x0)
    #     v2 = np.dot(grad1(x0), x - x0)
    #     v3 = 1 / (2 * alpha) * np.dot(x - x0, x - x0)
    #     return v1 + v2 + v3
    # return f
    g = grad1(x0)
    return x0 - alpha * g

def check_grad():
    x0 = np.random.normal(size=(N))
    f1 = func(x0)
    grad = grad_func(x0)
    dir = np.random.normal(size=(N))
    t = 1e-6
    x1 = x0 + t * dir
    f2 = func(x1)
    print(f2 - f1)
    print(t * np.dot(grad, dir))


def majorization(x, func, gradf, geng):
    f_list, x_list, g_list = [], [], []
    global alpha
    while True:
        grad = gradf(x)
        f = func(x)
        x_list.append(x)
        g_list.append(grad)
        f_list.append(f)
        
        if norm(grad) < ETA:
            break
        newx = geng(x)
        # res = optimize.minimize(g, x, method="BFGS")
        if norm(newx - x) < ETA:
            alpha = alpha / 2
        x = newx
        print(f)
        print(norm(grad))


    return f_list, x_list, g_list


x = -np.ones(N)
fl, xl, gl = majorization(x, func, grad_func, geng1)
# plt.figure(figsize=(6, 4))
# plt.plot(list(range(len(gl))), [v - P_STAR for v in fl])
# plt.xlabel("steps")
# plt.ylabel("f-p*")
# plt.title(f"Memory step: {s}")
# plt.savefig(f"lbfgs-{s}.png", bbox_inches="tight")
