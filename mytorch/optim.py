import numpy as np
from mytorch.tensor import Tensor


class Optimizer:

    def __init__(self, params: list[Tensor], lr: float = 1e-3) -> None:
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):

    # SGD with momentum
    def __init__(self, params: list[Tensor], lr: float = 1e-3,
                 gamma: float = 0.9) -> None:
        super().__init__(params, lr)
        self.gamma = gamma
        self.v = [np.zeros(param.shape) for param in self.params]

    def step(self) -> None:
        for i in range(len(self.params)):
            grad = self.params[i].grad
            self.v[i] = (self.gamma * self.v[i]) + (self.lr * grad)
            self.params[i].data = self.params[i].data - self.v[i]


class Adam(Optimizer):

    def __init__(self, params: list[Tensor], lr: float = 1e-3, b1: float = 0.9, 
                 b2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__(params, lr)
        self.b1 = b1
        self.b2 = b2
        # initialize first and second moment vectors
        self.m = [np.zeros(param.shape) for param in self.params]
        self.v = [np.zeros(param.shape) for param in self.params]
        self.t = 0  # timestep
        self.epsilon = epsilon

    def step(self) -> None:
        self.t += 1
        for i in range(len(self.params)):
            grad = self.params[i].grad
            self.m[i] = (self.b1 * self.m[i]) + ((1 - self.b1) * grad)
            self.v[i] = (self.b2 * self.v[i]) + \
                ((1 - self.b2) * np.power(grad, 2))
            m_hat = self.m[i] / (1 - np.power(self.b1, self.t))
            v_hat = self.v[i] / (1 - np.power(self.b2, self.t))
            self.params[i].data = self.params[i].data - \
                ((self.lr * m_hat) / (np.sqrt(v_hat) + self.epsilon))
