import numpy as np


class Function:
    def __init__(self) -> None:
        self.prev = []

    def save_for_backward(self, *tensors: 'Tensor') -> None:
        self.prev.extend(tensors)


class Add(Function):
    def __repr__(self) -> str:
        return f"Function(Add, prev={self.prev})"

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward(a, b)
        out = a.data + b.data
        return out

    def backward(self, out: 'np.ndarray') -> None:
        a, b = self.prev
        a.grad = out
        b.grad = out


class Dot(Function):
    def __repr__(self) -> str:
        return f"Function(Dot, prev={self.prev})"

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward(a, b)
        out = a.data.dot(b.data)
        return out

    def backward(self, out: 'np.ndarray') -> None:
        a, b = self.prev
        a.grad = out.dot(b.T.data)
        b.grad = a.T.data.dot(out)
