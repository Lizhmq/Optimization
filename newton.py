import numpy as np



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

def J(x1, x2):
    v11 = -20 * x1
    v12 = 10
    v21 = -1
    v22 = 0
    return np.array([[v11, v12], [v21, v22]])

def solve_linear(A, b):
    return np.linalg.solve(A, b)

def get_init():
    return -2, 2

ETA = 1e-5
P_STAR = 0