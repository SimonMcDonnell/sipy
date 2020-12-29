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
        return fn.forward(self, b)

    def __sub__(self, b: 'Tensor') -> 'Tensor':
        fn = Sub()
        return fn.forward(self, b)

    def dot(self, b: 'Tensor') -> 'Tensor':
        fn = Dot()
        return fn.forward(self, b)

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


class Function:
    def __init__(self) -> None:
        self.prev = []

    def __call__(self, *inputs: 'Tensor') -> None:
        # to be overwritten in every subclass
        raise NotImplementedError

    def save_for_backward(self, *tensors: 'Tensor') -> None:
        self.prev.extend(tensors)


class Add(Function):
    def __repr__(self) -> str:
        return f"Function(Add, prev={self.prev})"

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward(a, b)
        out = Tensor(a.data + b.data)
        out.grad_fn = self
        return out

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out
        b.grad = out


class Sub(Function):
    def __repr__(self) -> str:
        return f"Function(Sub, prev={self.prev})"

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward(a, b)
        out = Tensor(a.data - b.data)
        out.grad_fn = self
        return out

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out
        b.grad = out


class Dot(Function):
    def __repr__(self) -> str:
        return f"Function(Dot, prev={self.prev})"

    def __call__(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        return self.forward(a, b)

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_for_backward(a, b)
        out = Tensor(a.data.dot(b.data))
        out.grad_fn = self
        return out

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out.dot(b.T.data)
        b.grad = a.T.data.dot(out)
