"""
Runs the Rayleigh bandit experiment, right panel of Figure 2 of the paper.
"""

import argparse
import functools
import time

import numpy as np
from joblib import delayed, Parallel

import bandit_envs
import utils
from rayleigh_algorithms import RayTS, RayPHE, RayEVILL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=200, type=int)
    parser.add_argument('--n_repeats', default=100, type=int)
    parser.add_argument('--warm_up', default=2, type=int)
    parser.add_argument('-d', default=2, type=int)
    parser.add_argument('-k', default=2, type=int)
    parser.add_argument('-a', default=1, type=float)
    parser.add_argument('--output_name', default='', type=str)
    args = parser.parse_args()

    arms = np.array([[1, 0.99], [0.1, 0.05]])
    params = np.array([0.9, 0.85])

    env = bandit_envs.RayleighBandit(arms, params)

    alg_names = ['EVILL', 'TSL', 'PHE']
    alg_inits = [functools.partial(alg_init, a=args.a) for alg_init in
                 (RayEVILL, RayTS, RayPHE)]

    results = {}
    for alg_name, alg_init in zip(alg_names, alg_inits):
        print(f"Running {alg_name}...")
        start = time.time()
        results[alg_name] = Parallel(n_jobs=-1)(
            delayed(utils.run_algorithm)(env, alg_init, args.n, warm_up_steps=args.warm_up) for _ in
            range(args.n_repeats))
        print(" %.1f seconds" % (time.time() - start))
    print()

    for key, value in results.items():
        regrets = np.array(value)[:, -1]
        print(
            f"{key}: mean {np.mean(regrets):5.2f}, "
            f"sem {np.std(regrets) / np.sqrt(args.n_repeats):5.2f}, "
            f"(median {np.median(regrets):5.2f}, "
            f"min {np.min(regrets):5.2f}, "
            f"max {np.max(regrets):5.2f})")

    
    if args.output_name:
        np.savez(f'results/{args.output_name}.npz', results)
