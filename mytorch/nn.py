from mytorch.tensor import Tensor, Function
import numpy as np


class Relu(Function):
    def __repr__(self) -> str:
        return f"Function(ReLU, prev={self.prev})"

    def __call__(self, a: 'Tensor') -> 'Tensor':
        return self.forward(a)

    def forward(self, a: 'Tensor') -> 'Tensor':
        self.save_for_backward(a)
        out = a.data
        out[out < 0] = 0
        out = Tensor(out)
        out.grad_fn = self
        return out

    def backward(self, out: np.ndarray) -> None:
        a = self.prev
        a.data[a.data <= 0] = 0
        a.data[a.data > 0] = 1
        a.grad = np.multiply(a.data, out)


class CrossEntropyLoss(Function):
    def __repr__(self) -> None:
        return f"Function(CrossEntropyLoss, prev={self.prev})"

    def __call__(self, outputs: 'Tensor', labels: 'Tensor') -> 'Tensor':
        return self.forward(outputs, labels)

    def forward(self, outputs: 'Tensor', labels: 'Tensor') -> 'Tensor':
        self.save_for_backward(outputs, labels)
        outputs = outputs.data
        labels = labels.data
        softmax = (np.exp(outputs[range(outputs.shape[0]), labels])) / \
            np.sum(np.exp(outputs), axis=1)
        out = Tensor(np.mean(-np.log(softmax)))
        out.grad_fn = self
        return out

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        outputs, labels = a.data, b.data
        softmax = np.exp(outputs) / np.sum(np.exp(outputs),
                                           axis=1).reshape(-1, 1)
        softmax[range(len(labels)), labels] -= 1
        (1/outputs.shape[0]) * softmax
