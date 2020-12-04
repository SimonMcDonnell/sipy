import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray, prev: tuple = ()) -> None:
        self.data = data
        self.prev = set(prev)
        self.grad = None
        self.grad_fn = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> 'Tensor':
        def _backward():
            self.grad = out.grad.T

        out = Tensor(self.data.T, (self, None))
        out.grad_fn = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, b: 'Tensor') -> 'Tensor':
        def _backward():
            self.grad = out.grad
            b.grad = out.grad

        out = Tensor(self.data + b.data, (self, b))
        out.grad_fn = _backward
        return out

    def dot(self, b: 'Tensor') -> 'Tensor':
        def _backward():
            self.grad = out.grad.dot(b.T)
            b.grad = self.grad.T.dot(out.grad)

        out = Tensor(self.data.dot(b.data), (self, b))
        out.grad_fn = _backward
        return out
