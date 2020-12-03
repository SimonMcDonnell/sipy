import numpy as np
from mytorch.function import Function


class Tensor:
    def __init__(self, data, grad=None, grad_fn=None, is_leaf=True,
                 requires_grad=False):
        self.data = data
        self.grad = grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape

    def backward(self):
