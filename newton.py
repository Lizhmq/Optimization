import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import solve
import time

def func(x):
    x1, x2 = x
    return 100 * (x2 - x1 * x1) ** 2 + (1 - x1) ** 2

def grad_func(x):
    x1, x2 = x
    a = 400 * x1 * (x1 ** 2 - x2) + 2 * (x1 - 1)
    b = 200 * (x2 - x1 ** 2)
    return np.array([a, b])

def Hessian(x):
    x1, x2 = x
    x11 = 1200 * x1 ** 2 - 400 * x2 + 2
    x12 = -400 * x1
    x21 = x12
    x22 = 200
    return np.array([[x11, x12], [x21, x22]])

def r(x):
    x1, x2 = x
    v1 = 10 * (x2 - x1 * x1)
    v2 = 1 - x1
    return np.array([v1, v2])

def J(x):
    x1, x2 = x
    v11 = -20 * x1
    v12 = 10
    v21 = -1
    v22 = 0
    return np.array([[v11, v12], [v21, v22]])

def norm(x):
    return np.sqrt(np.sum(np.square(x)))

def solve_linear(A, b):
    return np.linalg.solve(A, b)

def get_init():
    return np.array([-2, 2])

x0 = get_init()
ETA = 1e-5
P_STAR = 0

def damped_newton(x, func, grad_func, alpha=0.5, beta=0.9):
    grad_list, step_list, f_list = [], [], []
    while True:
        grad = grad_func(x)
        dir = -solve_linear(Hessian(x), grad)
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


def gauss_newton(x, func, grad_func):
    grad_list, step_list, f_list = [], [], []
    while True:
        grad = grad_func(x)
        dir = -solve_linear(np.dot(J(x).T, J(x)), np.dot(J(x).T, r(x)))
        g_norm = norm(grad)
        f_ = func(x)
        f_list.append(f_)
        grad_list.append(g_norm)
        if g_norm <= ETA:       # stop criteria
            break
        t = 1
        x = x + t * dir
    
    return grad_list, step_list, f_list
    


# grid search
alphas = [0.5, 0.3, 0.1, 0.03][::-1]
betas = alphas
steps = np.zeros((len(alphas), len(betas)))
values = np.zeros((len(alphas), len(betas)))

for i in range(len(alphas)):
    for j in range(len(betas)):
        gl, sl, fl = damped_newton(x0, func, grad_func, alpha=alphas[i], beta=betas[j])
        steps[i, j] = len(gl)
        values[i, j] = fl[-1]

print(steps, values)

time0 = time.time()
alpha, beta = 0.03, 0.5
gl, sl, fl = damped_newton(x0, func, grad_func, alpha, beta)
time1 = time.time()

plt.figure(figsize=(6, 4))
plt.plot(list(range(len(gl))), [np.log(v) for v in fl])
plt.xlabel("steps")
plt.ylabel("log(f-p*)")
plt.savefig("dm-log(f-p).png", bbox_inches="tight")

time2 = time.time()
gl, sl, fl = gauss_newton(x0, func, grad_func)
time3 = time.time()

plt.figure(figsize=(6, 4))
plt.plot(list(range(len(gl))), [np.log(v) for v in fl])
plt.xlabel("steps")
plt.ylabel("log(f-p*)")
plt.savefig("gn-log(f-p).png", bbox_inches="tight")

print(f"Damped Newton time: {time1 - time0}s")
print(f"Gauss-Newton time: {time3 - time2}s")
