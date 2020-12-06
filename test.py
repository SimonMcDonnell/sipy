from mytorch.tensor import Tensor
import numpy as np

a = Tensor(np.array([[1, 2, 3, 4]]))
b = Tensor(np.array([[1], [2], [3], [4]]))
print(a.shape, b.shape)

c = a.dot(b)
print(c, c.shape)

c.backward()
print(a.grad, b.grad)
