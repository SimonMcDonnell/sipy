import numpy as np
from mytorch.tensor import Tensor, Function


###############################################################################
# Activation Functions
###############################################################################


class ReLU(Function):
    def __repr__(self) -> str:
        return f"Function(ReLU)"

    def forward(self, a: Tensor) -> Tensor:
        self.save_for_backward(a)
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

    def forward(self, a: Tensor) -> Tensor:
        self.save_for_backward(a)
        out = self.sigmoid(a.data)
        return Tensor(out, grad_fn=self, requires_grad=a.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        dx_sigmoid = np.multiply(
            self.sigmoid(a.data), (1-self.sigmoid(a.data))
        )
        a.grad = np.multiply(dx_sigmoid, out)


def relu(a: Tensor) -> Tensor:
    return ReLU()(a)

def sigmoid(a: Tensor) -> Tensor:
    return Sigmoid()(a)


###############################################################################
# Loss Functions
###############################################################################


class BCELoss(Function):

    def __repr__(self) -> None:
        return f"Function(BCELoss)"

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        self.save_for_backward(outputs, labels)
        output_data = outputs.data
        labels = labels.data
        loss = -(labels * np.log(output_data)) - \
            ((1-labels) * np.log(1-output_data))
        return Tensor(np.mean(loss), grad_fn=self,
                      requires_grad=outputs.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        outputs, labels = a.data, b.data
        a.grad = (-labels/outputs) + (1-labels)/(1-outputs)


class CrossEntropyLoss(Function):

    def __repr__(self) -> None:
        return f"Function(CrossEntropyLoss)"

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        self.save_for_backward(outputs, labels)
        output_data = outputs.data
        labels = labels.data
        softmax = (np.exp(output_data[range(output_data.shape[0]), labels])) / \
            np.sum(np.exp(output_data), axis=1)
        return Tensor(np.mean(-np.log(softmax)), grad_fn=self,
                      requires_grad=outputs.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        outputs, labels = a.data, b.data
        softmax = np.exp(outputs) / np.sum(np.exp(outputs),
                                           axis=1).reshape(-1, 1)
        softmax[range(len(labels)), labels] -= 1
        a.grad = (1/outputs.shape[0]) * softmax


def binary_cross_entropy(outputs: Tensor, labels: Tensor) -> Tensor:
    return BCELoss()(outputs, labels)

def cross_entropy(outputs: Tensor, labels: Tensor) -> Tensor:
    return CrossEntropyLoss()(outputs, labels)
