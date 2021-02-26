import numpy as np
from typing import List


class Tensor:
    def __init__(self, data: np.ndarray, grad_fn: 'Function' = None,
                 requires_grad: bool = False) -> None:
        self.data = data
        self.grad = None
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> 'Tensor':
        def _backward():
            self.grad = out.grad.T

        out = Tensor(self.data.T, grad_fn=_backward,
                     requires_grad=self.requires_grad)
        return out

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data}, grad_fn={self.grad_fn}, requires_grad={self.requires_grad})"

    def __add__(self, b: 'Tensor') -> 'Tensor':
        add = Add()
        return add(self, b)

    def __sub__(self, b: 'Tensor') -> 'Tensor':
        sub = Sub()
        return sub(self, b)

    def dot(self, b: 'Tensor') -> 'Tensor':
        dot_fn = Dot()
        return dot_fn(self, b)

    def backward(self) -> None:
        # build the graph
        graph, visited = [], set()

        def build_graph(node: 'Tensor'):
            if node not in visited and node.requires_grad is True:
                visited.add(node)
                if node.grad_fn:
                    for prev in node.grad_fn.prev:
                        build_graph(prev)
                graph.append(node)
        build_graph(self)

        # backpropagate gradient
        self.grad = np.array([1.]).reshape(1, 1)  # implicit gradient creation
        for node in reversed(graph):
            if node.grad_fn:
                node.grad_fn.backward(node.grad)


class Function:
    def __init__(self) -> None:
        self.prev = []

    def __call__(self, *inputs: 'Tensor') -> None:
        return self.forward(*inputs)

    def forward(self, *inputs: 'Tensor') -> None:
        raise NotImplementedError

    def save_for_backward(self, tensors: List['Tensor']) -> None:
        self.prev = tensors


class Add(Function):
    def __repr__(self) -> str:
        return f"Function(Add)"

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward([a, b])
        return Tensor(a.data + b.data, grad_fn=self,
                      requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out
        b.grad = out


class Sub(Function):
    def __repr__(self) -> str:
        return f"Function(Sub)"

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward([a, b])
        return Tensor(a.data - b.data, grad_fn=self,
                      requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out
        b.grad = out


class Dot(Function):
    def __repr__(self) -> str:
        return f"Function(Dot)"

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward([a, b])
        return Tensor(a.data.dot(b.data), grad_fn=self,
                      requires_grad=(a.requires_grad or b.requires_grad))

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out.dot(b.T.data)
        b.grad = a.T.data.dot(out)
