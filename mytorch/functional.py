from mytorch.tensor import Tensor, Function
import numpy as np


def relu(input: 'Tensor') -> 'Tensor':
    return Relu()(input)


def sigmoid(input: 'Tensor') -> 'Tensor':
    return Sigmoid()(input)


class Relu(Function):
    def __repr__(self) -> str:
        return f"Function(ReLU)"

    def forward(self, a: 'Tensor') -> 'Tensor':
        self.save_for_backward([a])
        out = a.data
        out[out < 0] = 0
        return Tensor(out, grad_fn=self, requires_grad=a.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        a.data[a.data <= 0] = 0
        a.data[a.data > 0] = 1
        a.grad = np.multiply(a.data, out)


class Sigmoid(Function):
    def __repr__(self) -> str:
        return f"Function(Sigmoid)"

    def sigmoid(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def forward(self, a: 'Tensor') -> 'Tensor':
        self.save_for_backward([a])
        out = self.sigmoid(a.data)
        return Tensor(out, grad_fn=self, requires_grad=a.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        dx_sigmoid = np.multiply(
            self.sigmoid(a.data), (1-self.sigmoid(a.data))
        )
        a.grad = np.multiply(dx_sigmoid, out)
