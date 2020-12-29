from mytorch.tensor import Tensor
from mytorch.nn import Relu, CrossEntropyLoss
import numpy as np

input = Tensor(np.random.randn(5, 20))
w1 = Tensor(np.random.randn(20, 15))
b1 = Tensor(np.random.randn(15,))
relu = Relu()
h1 = relu(input.dot(w1) + b1)

w2 = Tensor(np.random.randn(15, 10))
b2 = Tensor(np.random.randn(10,))
h2 = h1.dot(w2) + b2

cel = CrossEntropyLoss()
loss = cel(h2, Tensor(np.array([1, 2, 0, 4, 3])))

loss.backward()
print(loss)
