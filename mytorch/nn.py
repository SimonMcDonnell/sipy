import numpy as np
import mytorch.functional as F
from mytorch.tensor import Tensor, Function


###############################################################################
# High Level Neural Network Components
###############################################################################


class Module:

    def __init__(self):
        self._modules = []
        self._parameters = []

    def __call__(self, *inputs: Tensor) -> None:
        return self.forward(*inputs)

    def forward(self, inputs: Tensor) -> None:
        # to be overwritten in every subclass
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        return self._parameters

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Module):
            self._modules.append(value)
            self._parameters.extend(value.parameters())
        object.__setattr__(self, name, value)


class Linear(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weights = Tensor(np.random.randn(
            in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.random.randn(out_features,), requires_grad=True)
        self._parameters.extend([self.weights, self.bias])

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weights + self.bias
    

###############################################################################
# Activation Functions
###############################################################################


class ReLU(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor) -> Tensor:
        return F.relu(a)


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor) -> Tensor:
        return F.sigmoid(a)


###############################################################################
# Loss Functions
###############################################################################


class BCELoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return F.binary_cross_entropy(outputs, labels)


class CrossEntropyLoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(outputs, labels)
