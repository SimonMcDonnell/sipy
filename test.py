from mytorch.tensor import Tensor
from mytorch import nn
import numpy as np


class Model(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(5, 1)

    def forward(self, inputs):
        return self.fc1(inputs)


model = Model()

input = Tensor(np.random.randn(1, 5), requires_grad=False)
output = model(input)

output.backward()
