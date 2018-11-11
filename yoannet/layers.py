"""
Our neural nets will be made up of layers,
Each layer needs to pass its inputs forward and propagate gradients backward.
For example:

inputs -> Linear -> Tanh -> Linear -> output
"""
import numpy as np

from typing import Dict, Callable

from yoannet.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str: Tensor] = {}
        self.grads: Dict[str: Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """"
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """"
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Computes output = input @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, input_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(z) and z = a * x + b
        then dy/dx = df/dz * dz/dx = f'(z) * a
        and dy/da = df/dz * dz/da = f'(z) * x
        and dy/db = df/dz * dz/db = f'(z) 

        if y = f(z) and z = w @ x + b
        then dy/dx = df/dz @ dz/dx = f'(z) @ w.T
        and dy/dw = df/dz @ dz/dw = x.T @ f'(z)
        and dy/db = df/dz @ dz/db = f'(z) 
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = g(a) and a = f(z)
        then dy/dx = dg/da * da/dz = grad * f'(input)
        """
        return self.f_prime(self.inputs) * grad



def tanh(x: Tensor) -> Tensor:
        return np.tanh(x)
    
def tanh_prime(x: Tensor) -> Tensor:
        y = tanh(x)
        return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
