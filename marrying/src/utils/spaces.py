from abc import ABC

import numpy as np


class Space(ABC):
    def uniform(self, size, device):
        pass

    def normal(self, mean, std, size, device):
        pass

    def laplace(self, mean, std, size, device):
        pass

    def generalized_normal(self, mean, lbd, p, size, device):
        pass

    @property
    def dim(self):
        pass


class NBoxSpace(Space):
    def __init__(self, n=1, min_=-1, max_=1):
        self.n = n
        self.min_ = min_
        self.max_ = max_
        self.center = (min_ + max_) / 2

    @property
    def dim(self):
        return self.n

    def uniform(self, size=None):
        if isinstance(size, int):
            return np.random.rand(size)
        else:
            return np.random.rand()

    def scale(self, x):
        return (x - self.min_) / (self.max_ - self.min_)

    def unscale(self, u):
        return u * (self.max_ - self.min_) + self.min_
