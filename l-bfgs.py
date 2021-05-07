import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import flip

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_normal(size):
    return np.random.normal(size=size)

N = 4

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

ETA = 1e-5


def func(x):
    return np.sum(np.power(x - 2, 2))

def grad_func(x):
    return 2 * x - 4

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

def Rosen(x):
    alpha = 2
    res = 0
    assert(len(x) == N)
    for i in range(N // 2):
        res += alpha * np.power(x[2 * i] ** 2 - x[2 * i + 1], 2)
        res += np.power(x[2 * i] - 1, 2)
    return res

def grad_Rosen(x):
    alpha = 2
    x1 = x[::2]
    x2 = x[1::2]
    g1 = 2 * x1 - 2
    g2 = 4 * x1 * (np.power(x1, 2) - x2)
    g3 = 2 * (x2 - np.power(x1, 2))
    res = np.zeros(x.shape)
    res[::2] = g1 + alpha * g2
    res[1::2] = alpha * g3
    return res


def backtracking(func, gradf, x, dx, alpha=0.3, beta=0.3):
    t = 1.
    while True:
        f1 = func(x) + alpha * t * np.dot(gradf(x), dx)
        f2 = func(x + t * dx)
        if f2 <= f1:
            break
        t *= beta
    return t

# def backtracking(func, gradf, x, dx, c1=0.4, c2=0.9, beta=0.95):
#     t = 1.
#     while True:
#         f1 = func(x) + c1 * t * np.dot(gradf(x), dx)
#         f2 = func(x + t * dx)
#         v1 = np.dot(gradf(x + t * dx), dx)
#         v2 = c2 * np.dot(gradf(x), dx)
#         if f2 <= f1 and v1 >= v2:
#             break
#         t *= beta
#     return t

def memoryback(xl, gl, memstep, Hk0=np.identity(N)):
    memstep = min(len(xl) - 1, memstep)
    deltax = np.array(xl[1:]) - np.array(xl[:-1])
    deltag = np.array(gl[1:]) - np.array(gl[:-1])
    q = gl[-1]  # gk
    for i in range(1, memstep + 1):
        dx = deltax[-i]
        dg = deltag[-i]
        rhok_1 = 1 / (np.dot(dg, dx) + 1e-8)
        alphak_1 = rhok_1 * np.dot(dx, q)
        q = q - alphak_1 * dg
    if len(deltax) > 0 and np.dot(deltag[-1], deltag[-1]) > 1e-2:
        # p = np.dot(deltax[-1], deltag[-1]) / (np.dot(deltag[-1], deltag[-1]) + 1e-8) * q
        p = np.dot(deltax[-1], deltag[-1]) / np.dot(deltag[-1], deltag[-1]) * q
    else:
        p = q
    for i in range(memstep, 0, -1):
        dx = deltax[-i]
        dg = deltag[-i]
        rhok_1 = 1 / (np.dot(dg, dx) + 1e-8)
        alphak_1 = rhok_1 * np.dot(dg, dx)
        betak_1 = rhok_1 * np.dot(dg, dx)
        p = p + (alphak_1 - betak_1) * dx
    return p    


def l_bfgs(x, func, grad_func, memstep, Hk=np.identity(2)):
    grad_list, f_list = [], []
    d_list, x_list = [], []
    while True:
        f = func(x)
        gk = grad_func(x)

        grad_list.append(gk)
        f_list.append(f)
        x_list.append(x)

        print(f)

        dk = -memoryback(x_list, grad_list, memstep)

        d_list.append(dk)

        g_norm = norm(gk)
        if g_norm <= ETA:
            break
        alpha = backtracking(func, grad_func, x, dk)
        x = x + alpha * dk

    return grad_list, f_list, x_list



# x1 = get_init(0)
# x2 = get_init(1)
# gl, fl, xl, Hl = dfp(x1, func1, grad_func1)
# print(len(gl))
# print(xl)

# gl, fl, xl, Hl = dfp(x2, func1, grad_func1)
# print(len(gl))
# print(xl)
P_STAR = 0

for s in [1, 5, 10, 30]:
    x = -np.ones(4)
    gl, fl, xl = l_bfgs(x, Rosen, grad_Rosen, memstep=s, Hk=np.identity(3))
    print(len(xl))
    # plt.figure(figsize=(6, 4))
    # plt.plot(list(range(len(gl))), [v - P_STAR for v in fl])
    # plt.xlabel("steps")
    # plt.ylabel("f-p*")
    # plt.title(f"Memory step: {s}")
    # plt.savefig(f"lbfgs-{s}.png", bbox_inches="tight")
