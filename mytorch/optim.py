from mytorch.tensor import Tensor, Function
from typing import List
import numpy as np


class SGD:
    # SGD with momentum
    def __init__(self, parameters: List['Tensor'], lr: float = 0.01,
                 gamma: float = 0.9) -> None:
        self.parameters = parameters
        self.lr = lr
        self.gamma = gamma
        self.v_prev = [np.zeros(param.shape) for param in self.parameters]

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = 0.0

    def step(self) -> None:
        for i in range(len(self.parameters)):
            v_t = (self.gamma * self.v_prev[i]) + \
                (self.lr * self.parameters[i].grad)
            self.parameters[i].data = self.parameters[i].data - v_t
            self.v_prev[i] = v_t
