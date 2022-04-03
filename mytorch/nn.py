from mytorch.tensor import Tensor, Function
from typing import List
import numpy as np


###############################################################################
# High Level Neural Network Components
###############################################################################

class Module:
    def __init__(self):
        self._modules = []
        self._parameters = []

    def __call__(self, inputs: 'Tensor') -> None:
        return self.forward(inputs)

    def forward(self, inputs: 'Tensor') -> None:
        # to be overwritten in every subclass
        raise NotImplementedError

    def parameters(self) -> List['Tensor']:
        return self._parameters

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Module):
            self._modules.append(value)
            self._parameters.extend(value.parameters())
        object.__setattr__(self, name, value)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.weights = Tensor(np.random.randn(
            in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.random.randn(out_features,), requires_grad=True)
        self._parameters.extend([self.weights, self.bias])

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        return inputs.dot(self.weights) + self.bias


###############################################################################
# Loss Functions
###############################################################################

class BCELoss(Function):
    def __repr__(self) -> None:
        return f"Function(BCELoss)"

    def forward(self, outputs: 'Tensor', labels: 'Tensor') -> 'Tensor':
        self.save_for_backward([outputs, labels])
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

    def forward(self, outputs: 'Tensor', labels: 'Tensor') -> 'Tensor':
        self.save_for_backward([outputs, labels])
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
