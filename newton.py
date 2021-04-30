import numpy as np



def func(x1, x2):
    return 100 * (x2 - x1 * x1) ** 2 + (1 - x1) ** 2

def grad_func(x1, x2):
    a = 400 * x1 * (x1 ** 2 - x2) + 2 * (x1 - 1)
    b = 200 * (x2 - x1 ** 2)
    return a, b

def Hessian(x1, x2):
    x11 = 1200 * x1 ** 2 - 400 * x2 + 2
    x12 = -400 * x1
    x21 = x12
    x22 = 200
    return np.array([[x11, x12], [x21, x22]])



def get_init():
    return -2, 2

ETA = 1e-5
P_STAR = 0