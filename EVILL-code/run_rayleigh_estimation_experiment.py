"""
Runs the Rayleigh estimation experiment, left panel of Figure 2 of the paper.
"""


import argparse
import functools

import numpy as np

import utils
from rayleigh_algorithms import rayleigh_mle, sigma


def rayleigh_phe(theta, features, targets, a=1.0):
    Z = a * np.random.normal(size=len(targets))
    return rayleigh_mle(theta, features, targets + Z)


def rayleigh_evill(theta, features, targets, a=1.0):
    n, d = features.shape
    W = a * np.sqrt(n) * np.random.normal(size=d)
    return rayleigh_mle(theta, features, targets, W)


def run_estimation_experiment(theta, estimate_fn, n, interval):
    arms = np.array([[1, 0], [0, 1]])
    features = arms[np.random.choice(2, size=n)]
    targets = np.random.rayleigh(sigma(features @ theta))
    theta_estimate = np.ones(2)

    errors = []
    for step in range(interval, n + 1, interval):
        theta_estimate = estimate_fn(theta_estimate, features[:step], targets[:step])
        errors.append(np.linalg.norm(theta_estimate - theta))

    return np.array(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=10000, type=int)
    parser.add_argument('-a', default=1.0, type=float)
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--n_repeats', default=100, type=int)
    parser.add_argument('--output_name', default='', type=str)
    args = parser.parse_args()

    theta = np.array([0.9, 0.85])

    print("running MLE")
    results_mle = np.array(
        [run_estimation_experiment(theta, rayleigh_mle, args.n, args.interval) for _ in
         range(args.n_repeats)])
    print("running PHE")
    results_phe = np.array([
        run_estimation_experiment(theta, functools.partial(rayleigh_phe, a=args.a), args.n,
                                  args.interval) for _ in range(args.n_repeats)])
    print("running EVILL")
    results_evill = np.array([
        run_estimation_experiment(theta, functools.partial(rayleigh_evill, a=args.a), args.n,
                                  args.interval) for _ in range(args.n_repeats)])

    print(
        f"MLE: {np.mean(results_mle, axis=0)[-1]:.3f} +/- {utils.sem(results_mle, axis=0)[-1]:.3f}")
    print(
        f"PHE: {np.mean(results_phe, axis=0)[-1]:.3f} +/- {utils.sem(results_phe, axis=0)[-1]:.3f}")
    print(
        f"EVILL: {np.mean(results_evill, axis=0)[-1]:.3f} +/- {utils.sem(results_evill, axis=0)[-1]:.3f}")

    x_vals = np.array(list(range(args.interval, args.n + 1, args.interval)))
    data = {'MLE': results_mle, 'PHE': results_phe, 'EVILL': results_evill, 'x_values': x_vals}

    
    if args.output_name:
        np.savez(f'results/{args.output_name}.npz', data)
