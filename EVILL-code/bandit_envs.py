from abc import ABC, abstractmethod

import numpy as np


class AbstractBanditEnv(ABC):
    def __init__(self, arms, parameter):
        self.arms = np.copy(arms)
        self.k, self.d = self.arms.shape
        self.parameter = np.copy(parameter)
        self.arm_param_product = self.arms @ self.parameter

    @property
    @abstractmethod
    def means(self):
        raise NotImplementedError

    @property
    def best_arm(self):
        return np.argmax(self.means)

    def sample_rewards(self):
        raise NotImplementedError

    def pull_arm(self, arm):
        reward = self.sample_rewards()[arm]
        best_arm = np.argmax(self.means)
        instant_pseudoregret = self.means[best_arm] - self.means[arm]
        return arm, reward, instant_pseudoregret


class LogisticBandit(AbstractBanditEnv):
    @property
    def means(self):
        return 1 / (1 + np.exp(-self.arm_param_product))

    def sample_rewards(self):
        return (np.random.rand(len(self.arms)) < self.means).astype(float)


class RayleighBandit(AbstractBanditEnv):
    @property
    def means(self):
        return np.sqrt(np.pi / 2) * self.sigma

    def sample_rewards(self):
        return np.random.rayleigh(scale=self.sigma)

    @property
    def sigma(self):
        return np.sqrt(1 / (2 * self.arm_param_product))
