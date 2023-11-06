import numpy as np
from scipy import optimize


def mean(x):
    return 1 / np.clip(x, 1e-6, None)


def variance(x):
    return 1 / (x ** 2)


def sigma(x):
    return np.sqrt(1 / (2 * x))


def likelihood_function(theta, features, targets, perturbation, regulariser=0.0):
    feat_theta_prod = features @ theta
    likelihood = np.sum(
        (targets ** 2) * feat_theta_prod - np.log(np.clip(feat_theta_prod, 1e-6, None)))

    return likelihood + np.inner(theta, perturbation) + regulariser / 2 * theta @ theta


def likelihood_gradient(theta, features, targets, perturbation, regulariser = 0.0):
    gradient = (targets ** 2 - mean(features @ theta)).T @ features
    return gradient + perturbation + regulariser * theta


def rayleigh_mle(theta, features, targets, perturbation=None, regulariser=0.0):
    if perturbation is None:
        perturbation = np.zeros_like(theta)

    result = optimize.minimize(likelihood_function, x0=theta,
                               args=(features, targets, perturbation, regulariser), jac=likelihood_gradient,
                               method='L-BFGS-B')
    return result.x


class RayBanditAlg:
    def __init__(self, arms, n, a, regulariser=1.0):
        self.arms = np.copy(arms)
        self.K, self.d = self.arms.shape
        self.n = n
        self.a = a
        self.regulariser = regulariser

        self.theta = np.ones(self.d)
        self.theta_tilde = np.ones(self.d)

        # sufficient statistics
        self.num_pulls = np.zeros(self.K)
        self.features = np.zeros((self.n, self.d))
        self.targets = np.zeros(self.n)

        self.arm_outer_prods = np.zeros((self.K, self.d, self.d))  # outer products of arm features
        for k in range(self.K):
            self.arm_outer_prods[k, :, :] = np.outer(self.arms[k, :], self.arms[k, :])

    def update(self, t, arm, r):
        self.features[t] = self.arms[arm]
        self.targets[t] = r
        self.num_pulls[arm] += 1

    def likelihood_hessian(self, theta):
        arms_theta_prod = self.arms @ theta
        hessian = (np.tensordot(variance(arms_theta_prod) * self.num_pulls, self.arm_outer_prods,
                                 axes=([0], [0])) + self.regulariser * np.eye(self.d))
        return hessian


class RayTS(RayBanditAlg):
    def get_arm(self, t):
        self.theta = rayleigh_mle(self.theta, self.features[:t], self.targets[:t], regulariser=self.regulariser)
        gram = self.likelihood_hessian(self.theta)
        gram_inv = np.linalg.inv(gram)

        theta_tilde = self.a * np.random.multivariate_normal(self.theta, gram_inv)
        return np.argmin(self.arms @ theta_tilde)


class RayPHE(RayBanditAlg):
    def get_arm(self, t):
        self.theta = rayleigh_mle(self.theta, self.features[:t], self.targets[:t], regulariser=self.regulariser)
        tf_prod = self.features[:t] @ self.theta
        noise_scale = self.a * np.sqrt(variance(tf_prod))
        Z = np.random.normal(scale=noise_scale)
        W = np.zeros(self.d)
        self.theta_tilde = rayleigh_mle(self.theta_tilde, self.features[:t], self.targets[:t] + Z,
                                        W, regulariser=self.regulariser)
        return np.argmin(self.arms @ self.theta_tilde)


class RayEVILL(RayBanditAlg):
    def get_arm(self, t):
        self.theta = rayleigh_mle(self.theta, self.features[:t], self.targets[:t], regulariser=self.regulariser)
        tf_prod = self.features[:t] @ self.theta
        noise_scale = self.a * np.sqrt(variance(tf_prod))
        Z = np.random.normal(scale=noise_scale, size=t)
        W = Z @ self.features[:t] +  np.sqrt(self.regulariser) * self.a * np.random.normal(size=self.d)
        
        self.theta_tilde = rayleigh_mle(self.theta_tilde, self.features[:t], self.targets[:t], W, regulariser=self.regulariser)
        return np.argmin(self.arms @ self.theta_tilde)
