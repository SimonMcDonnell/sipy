import numpy as np
from mytorch.tensor import Tensor


class Function:
    def __init__(self, )


class Dot(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data.dot(b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output.dot(b.data.T)
        grad_b = a.data.T.dot(grad_output)
        return grad_a, grad_b


class MSSE(Function):
    @staticmethod
    def forward(ctx, )
