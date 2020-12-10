from mytorch.function import Add, Dot, Relu
import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad = None
        self.grad_fn = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> 'Tensor':
        def _backward():
            self.grad = out.grad.T

        out = Tensor(self.data.T)
        out.grad_fn = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, b: 'Tensor') -> 'Tensor':
        fn = Add()
        out = Tensor(fn.forward(self, b))
        out.grad_fn = fn
        return out

    def dot(self, b: 'Tensor') -> 'Tensor':
        fn = Dot()
        out = Tensor(fn.forward(self, b))
        out.grad_fn = fn
        return out

    def relu(self) -> 'Tensor':
        fn = Relu()
        out = Tensor(fn.forward(self))
        out.grad_fn = fn
        return out

    def backward(self) -> None:
        # build the graph
        graph, visited = [], set()

        def build_graph(node: 'Tensor'):
            if node not in visited:
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
