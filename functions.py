import numpy as np


def choose_fun(flag_fun, flag_scope=True):
    match flag_fun:
        case 1:
            choose = rastrigin_function, (-5.12, 5.12)
        case 2:
            choose = rosenbrock_function, (-2048, 2048)
        case 3:
            choose = sphere_function, (-100, 100)
        case 4:
            choose = step_function, (-10, 10)
        case 5:
            choose = sum_power_function, (-1, 1)
        case 6:
            choose = schwefel_function, (-10, 10)
        case _:
            raise ValueError("Wrong function number")
    return choose if flag_scope else choose[0]


def rastrigin_function(x):
    return 10 * len(x) + sum(
        x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x)))


def rosenbrock_function(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in
               range(len(x) - 1))


def sphere_function(x):
    return sum(i ** 2 for i in x)


def step_function(x):
    return sum((i + 0.5) ** 2 for i in x)


def sum_power_function(x):
    return sum(np.abs(x[i]) ** (i + 1) for i in range(len(x)))


def schwefel_function(x):
    return 418.9829 * len(x) - sum(i * np.sin(np.sqrt(np.abs(i))) for i in x)
