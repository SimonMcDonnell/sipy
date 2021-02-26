from mytorch.tensor import Tensor, Function
from typing import List
import numpy as np


class SGD:
    def __init__(self, parameters: List['Tensor'], lr: float = 0.01) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = 0.0

    def step(self) -> None:
        for param in self.parameters:
            param.data = param.data - self.lr * param.grad
