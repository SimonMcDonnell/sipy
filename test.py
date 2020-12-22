from mytorch.tensor import Tensor
from mytorch.nn import Relu
import numpy as np

a = Tensor(np.array([[1, 2, 3, 4]]))
b = Tensor(np.array([[1, -5], [2, 6], [3, -7], [4, -8]]))
print(a.shape, b.shape)

c = a.dot(b)
print(c, c.shape)

relu = Relu()
out = relu(c)
print(out)
