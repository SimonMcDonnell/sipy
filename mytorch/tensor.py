import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray, prev: tuple = ()) -> None:
        self.data = data
        self.prev = set(prev)
        self.grad = None
        self.grad_fn = lambda: None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> 'Tensor':
        def _backward():
            self.grad = out.grad.T

        out = Tensor(self.data.T, (self,))
        out.grad_fn = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, b: 'Tensor') -> 'Tensor':
        def _backward():
            self.grad = out.grad
            b.grad = out.grad

        out = Tensor(self.data + b.data, (self, b))
        out.grad_fn = _backward
        return out

    def dot(self, b: 'Tensor') -> 'Tensor':
        def _backward():
            self.grad = out.grad.dot(b.T.data)
            b.grad = self.grad.T.dot(out.grad)

        out = Tensor(self.data.dot(b.data), (self, b))
        out.grad_fn = _backward
        return out

    def relu(self) -> 'Tensor':
        def _backward():
            self.grad = (out.data > 0) * out.grad

        out_data = self.data
        out_data[out_data < 0] = 0
        out = Tensor(out_data, (self,))
        out.grad_fn = _backward
        return out

    def backward(self) -> None:
        # build the graph
        graph, visited = [], set()

        def build_graph(node: 'Tensor'):
            if node not in visited:
                visited.add(node)
                for prev in node.prev:
                    build_graph(prev)
                graph.append(node)
        build_graph(self)

        # backpropagate gradient
        self.grad = np.array([1.]).reshape(1, 1)  # implicit gradient creation
        for node in reversed(graph):
            node.grad_fn()
