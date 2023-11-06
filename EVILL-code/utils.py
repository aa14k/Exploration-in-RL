import itertools

import numpy as np
from scipy import optimize


def sem(v, axis=0):
    v = np.array(v)
    stds = np.std(v, axis=axis)
    l = v.shape[axis]
    return stds / np.sqrt(l)


def generate_arms(d, s):
    """Genertes d-dimensional arms that are s-hot vectors"""
    return np.array([[1 if i in indices else 0 for i in range(d)] for indices in
                     itertools.combinations(range(d), s)])


def generate_instance_params(d, s, decay, lin_val_min, lin_val_max):
    def func(c):
        c1, c2 = c
        return (np.sum([(c1 / (i + 1 + c2)) ** decay for i in range(s)]) - lin_val_max,
                np.sum([(c1 / (d - i + c2)) ** decay for i in range(s)]) - lin_val_min)

    c1, c2 = optimize.fsolve(func, [2, 1])
    params = np.array([(c1 / (c2 + i + 1)) ** decay for i in range(d)])
    return params


def run_algorithm(env, alg_init, n, warm_up_steps=0, period_size=1):
    alg = alg_init(env.arms, n)
    regret = np.zeros(n // period_size)
    for step in range(n):
        if step < warm_up_steps:
            arm = np.random.randint(env.k)
        else:
            arm = alg.get_arm(step)
        _, reward, instant_pseudoregret = env.pull_arm(arm)
        alg.update(step, arm, reward)
        regret[step // period_size] += instant_pseudoregret

    return np.cumsum(regret)
