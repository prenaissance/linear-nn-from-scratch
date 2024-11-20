from typing import Generator, List

import numpy as np
from numpy.typing import NDArray

from .base import Module, Parameter
from .functional import relu, sigmoid
from scipy.special import softmax


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(shape=(out_features, in_features))
        self.bias = Parameter(shape=(out_features, 1))

    def forward(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        # Wrap input if it's a single sample
        if input.ndim == 1:
            input = input[:, np.newaxis]
        return self.weight.data @ input + self.bias.data

    def backward(
        self, input: NDArray[np.float32], loss_grad: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        self.weight.grad += loss_grad @ input.T / input.shape[0]
        self.bias.grad += np.sum(loss_grad, axis=1, keepdims=True) / input.shape[0]
        return self.weight.data.T @ loss_grad


class Sigmoid(Module):
    def forward(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        return sigmoid(input)

    def backward(
        self, input: NDArray[np.float32], loss_grad: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        return loss_grad * sigmoid(input) * (1 - sigmoid(input)) / input.shape[0] ** 2


# class ReLU(Module):
#     @staticmethod
#     def _relu_derivative(x: NDArray[np.float32]) -> NDArray[np.float32]:
#         derivatives = (x > 0).astype(np.float32)
#         return derivatives

#     def forward(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
#         return relu(input)

#     def backward(
#         self, input: NDArray[np.float32], loss_grad: NDArray[np.float32]
#     ) -> NDArray[np.float32]:
#         # return self._relu_derivative(loss_grad)
#         return loss_grad


class Softmax(Module):
    def forward(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        return softmax(input)

    def backward(
        self, input: NDArray[np.float32], loss_grad: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        return loss_grad


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = layers
        self._inputs: List[NDArray[np.float32]] = []

    def forward(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        inputs: List[NDArray[np.float32]] = []
        for layer in self.layers:
            inputs.append(input)
            input = layer(input)
        self._inputs = inputs
        return input

    def backward(
        self, input: NDArray[np.float32], loss_grad: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        for layer, prev_input in zip(reversed(self.layers), reversed(self._inputs)):
            loss_grad = layer.backward(prev_input, loss_grad)
        self._inputs = []
        return loss_grad

    def parameters(self) -> Generator[Parameter, None, None]:
        for layer in self.layers:
            yield from layer.parameters()
