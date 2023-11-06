import numpy as np


def clipped_sigmoid(x):
    x[x > 36] = 36
    x[x < -36] = 36
    return 1 / (1 + np.exp(-x))


def variance(x):
    return clipped_sigmoid(x) * (1 - clipped_sigmoid(x))


def irls(theta, arms, arm_outer_prods, num_pos, num_neg, regularisation=1.0, pre_perturbation=0.0,
         post_perturbation=0.0, num_iter=1000, tol=1e-8):
    """
    Iterative reweighted least squares for Bayesian logistic regression. See Sections 4.3.3 and
     4.5.1 in
        Bishop, Christopher M., and Nasser M. Nasrabadi. Pattern Recognition and Machine Learning.
        Vol. 4. No. 4. New York: Springer, 2006.

    Returns: estimate of parameters and gram matrix
    """
    arms = np.copy(arms)
    theta = np.copy(theta)
    _, d = arms.shape
    gram = np.eye(d)

    for i in range(num_iter):
        theta_old = np.copy(theta)

        arms_theta_prod = arms @ theta
        means = clipped_sigmoid(arms_theta_prod)
        num_pulls = num_pos + num_neg
        gram = (np.tensordot(variance(arms_theta_prod) * num_pulls, arm_outer_prods,
                             axes=([0], [0])) + regularisation * np.eye(d))
        Rz = variance(
            arms_theta_prod) * num_pulls * arms_theta_prod + num_pos - num_pulls * means + pre_perturbation
        theta = np.linalg.solve(gram, arms.T @ Rz + post_perturbation)

        if np.linalg.norm(theta - theta_old) < tol:
            break
    return theta, gram


class LogBanditAlg:
    def __init__(self, arms, n, a):
        self.arms = np.copy(arms)
        self.K, self.d = self.arms.shape
        self.n = n
        self.a = a

        self.theta = np.zeros(self.d)

        # sufficient statistics
        self.features = np.zeros((n, self.d))
        self.obs = np.zeros(self.n)
        self.num_pulls = np.zeros(self.K)
        self.num_pos = np.zeros(self.K, dtype=int)  # number of positive observations
        self.num_neg = np.zeros(self.K, dtype=int)  # number of negative observations
        self.arm_outer_prods = np.zeros((self.K, self.d, self.d))  # outer products of arm features

        for k in range(self.K):
            self.arm_outer_prods[k, :, :] = np.outer(self.arms[k, :], self.arms[k, :])

    def update(self, t, arm, reward):
        self.features[t] = self.arms[arm]
        self.obs[t] = reward
        self.num_pos[arm] += reward
        self.num_neg[arm] += 1 - reward
        self.num_pulls[arm] += 1

    def get_arm(self, t):
        raise NotImplementedError()


class LogTS(LogBanditAlg):
    def get_arm(self, t):
        self.theta, gram = irls(self.theta, self.arms, self.arm_outer_prods, self.num_pos,
                                self.num_neg, regularisation=1.0)
        gram_inv = np.linalg.inv(gram)

        theta_tilde = self.a * np.random.multivariate_normal(self.theta, gram_inv)
        mu = self.arms @ theta_tilde
        return np.argmax(mu)


class LogFPL(LogBanditAlg):
    def get_arm(self, t):
        """The factor of 1/2 on the following line corresponds to square root of the worst-case
        variance for the logistic bandit. This brings the effective value of a for this algorithm
        in line with that used by LogTS and LogPHE, in the worst case."""
        z = self.a / 2 * np.sqrt(self.num_pulls) * np.random.randn(self.K)
        y = np.random.randn(self.d) * self.a

        theta, _ = irls(self.theta, self.arms, self.arm_outer_prods, self.num_pos, self.num_neg,
                        regularisation=1.0, pre_perturbation=z, post_perturbation=y)

        mu = clipped_sigmoid(self.arms @ theta)
        return np.argmax(mu)


class LogPHE(LogBanditAlg):
    def get_arm(self, t):
        mean_theta, _ = irls(self.theta, self.arms, self.arm_outer_prods, self.num_pos,
                             self.num_neg, regularisation=1.0)

        u = self.arms @ mean_theta
        z = self.a * np.sqrt(variance(u)) * np.sqrt(self.num_pulls) * np.random.randn(self.K)
        y = np.random.randn(self.d) * self.a

        theta, _ = irls(self.theta, self.arms, self.arm_outer_prods, self.num_pos, self.num_neg,
                        regularisation=1.0, pre_perturbation=z, post_perturbation=y)

        mu = clipped_sigmoid(self.arms @ theta)
        return np.argmax(mu)
