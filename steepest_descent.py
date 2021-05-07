import random
import numpy as np
from matplotlib import pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

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


def check_grad():
    x0 = get_init((5, ))
    a = get_normal((N, M))
    func = lambda x: func1(a, x)
    grad_func = lambda x: grad_func1(a, x)
    f1 = func(x0)
    grad = grad_func(x0)
    dir = np.random.normal(size=(5))
    t = 1e-6
    x1 = x0 + t * dir
    f2 = func(x1)
    print(f2 - f1)
    print(t * np.dot(grad, dir))



M, N = 10, 5
ETA = 1e-5
P_STAR = 2 * M
alpha, beta = None, None

def steepest_descent(x, func, grad_func, alpha=0.5, beta=0.9):
    grad_list, step_list, f_list = [], [], []
    while True:
        grad = grad_func(x)
        dir = -np.sign(grad)
        g_norm = norm(grad)
        if g_norm <= ETA:       # stop criteria
            break
        f_ = func(x)
        f_list.append(f_)
        grad_list.append(g_norm)
        t = 1
        while True:             # backtracking line search
            f2 = func(x + t * dir)
            if f2 <= f_ + alpha * t * np.dot(grad, dir):
                break
            t *= beta
        step_list.append(t)
        x = x + t * dir
    
    return grad_list, step_list, f_list
    



set_seed(2233)

x0 = get_init((N, ))
a = get_normal((N, M))
func = lambda x: func1(a, x)
grad_func = lambda x: grad_func1(a, x)


# grid search
alphas = [0.5, 0.3, 0.1, 0.03][::-1]
betas = alphas
steps = np.zeros((len(alphas), len(betas)))
values = np.zeros((len(alphas), len(betas)))

for i in range(len(alphas)):
    for j in range(len(betas)):
        gl, sl, fl = steepest_descent(x0, func, grad_func, alpha=alphas[i], beta=betas[j])
        steps[i, j] = len(gl)
        values[i, j] = fl[-1]

print(steps, values)

alpha, beta = 0.1, 0.5
gl, sl, fl = steepest_descent(x0, func, grad_func, alpha, beta)
plt.figure(figsize=(6, 4))
plt.plot(list(range(len(gl))), gl)
plt.xlabel("steps")
plt.ylabel("gradient norm")
plt.savefig("st-gradient.png", bbox_inches="tight")

plt.figure(figsize=(6, 4))
plt.plot(list(range(len(gl))), sl)
plt.xlabel("steps")
plt.ylabel("step length")
plt.savefig("st-step_l.png", bbox_inches="tight")

plt.figure(figsize=(6, 4))
plt.plot(list(range(len(gl))), [v - P_STAR for v in fl])
plt.xlabel("steps")
plt.ylabel("f-p*")
plt.savefig("st-f-p.png", bbox_inches="tight")
