"""
Runs the logistic bandit experiment featuring in Figure 1 of the paper.
"""

import argparse
import functools

import numpy as np
from joblib import Parallel, delayed

import bandit_envs
import logistic_algorithms
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=5000, type=int)
    parser.add_argument('-d', default=10, type=int)
    parser.add_argument('-s', default=3, type=int)
    parser.add_argument('-a', default=1.0, type=float)
    parser.add_argument('--n_repeats', default=100, type=int)
    parser.add_argument('--warm_up', default=120, type=int)
    parser.add_argument('--lin_val_max', default=3.0, type=float)
    parser.add_argument('--lin_val_min', default=0.25, type=float)
    parser.add_argument('--param_decay', default=1.0, type=float)
    parser.add_argument('--output_name', default='', type=str)
    args = parser.parse_args()

    arms = utils.generate_arms(args.d, args.s)
    np.random.shuffle(arms)
    print(f"Some example arms:")
    print(arms[0])
    print(arms[5])
    print(arms[10])
    print()

    params = utils.generate_instance_params(args.d, args.s, decay=args.param_decay,
                                            lin_val_min=args.lin_val_min,
                                            lin_val_max=args.lin_val_max)
    env = bandit_envs.LogisticBandit(arms, params)

    lin_prod = arms @ params
    print(
        f"Param stats: max linear value {max(lin_prod):.3f}, min linear value {min(lin_prod):.3f}")
    print()

    alg_names = ['TSL', 'FPL', 'PHE']
    alg_inits = [functools.partial(alg_init, a=args.a) for alg_init in (
        logistic_algorithms.LogTS, logistic_algorithms.LogFPL, logistic_algorithms.LogPHE)]

    results = {}
    _run_alg = lambda alg_init: delayed(utils.run_algorithm)(env, alg_init, args.n, warm_up_steps=args.warm_up)
    for alg_name, alg_init in zip(alg_names, alg_inits):
        print(f"Running {alg_name}...")
        results[alg_name] = Parallel(n_jobs=-1)(_run_alg(alg_init) for _ in range(args.n_repeats))
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
