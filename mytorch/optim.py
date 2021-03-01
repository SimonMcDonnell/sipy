from mytorch.tensor import Tensor, Function
from typing import List
import numpy as np


class Optimizer:
    def __init__(self, params: List['Tensor'], lr: float = 1e-3) -> None:
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    # SGD with momentum
    def __init__(self, params: List['Tensor'], lr: float = 1e-3,
                 gamma: float = 0.9) -> None:
        super(SGD, self).__init__(params, lr)
        self.gamma = gamma
        self.v_prev = [np.zeros(param.shape) for param in self.params]

    def step(self) -> None:
        for i in range(len(self.params)):
            v_t = (self.gamma * self.v_prev[i]) + \
                (self.lr * self.params[i].grad)
            self.params[i].data = self.params[i].data - v_t
            self.v_prev[i] = v_t
