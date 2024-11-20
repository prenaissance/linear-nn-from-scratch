from abc import ABC, abstractmethod
from typing import Iterable, List

from .base import Parameter


class Optimizer(ABC):
    parameters: List[Parameter]

    def __init__(self, parameters: Iterable[Parameter]):
        self.parameters = list(parameters)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    @abstractmethod
    def step(self):
        raise NotImplementedError


class GDOptimizer(Optimizer):
    """
    Simple Gradient Descent optimizer. No momentum, no learning rate decay, no randomness.
    """

    def __init__(self, parameters: Iterable[Parameter], lr: float):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad
