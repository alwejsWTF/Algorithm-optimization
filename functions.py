import numpy as np

def trid_function(x):
    sum_squares = np.sum((x - 1)**2)
    sum_product = np.sum(x[:-1] * x[1:])
    return sum_squares - sum_product


def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rastrigin_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def sphere_function(x):
    return np.sum(x ** 2)