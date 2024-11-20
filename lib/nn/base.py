from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray


class Parameter:
    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        data: Optional[NDArray[np.float32]] = None,
    ):
        if data is None and shape is None:
            raise ValueError("Either shape or data must be provided")

        self.data = (
            data
            if data is not None
            else cast(NDArray[np.float32], np.random.randn(*shape)).astype(np.float32)
        )
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad.fill(0)

    def __repr__(self):
        return f"Parameter<{self.data.shape}>()"

    def __str__(self):
        return f"Parameter<{self.data.shape}>()"

    def add_(self, other: "Parameter"):
        self.data += other.data
        self.grad += other.grad
        return self


class Module(ABC):
    @abstractmethod
    def forward(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        pass

    @abstractmethod
    def backward(
        self, input: NDArray[np.float32], loss_grad: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def parameters(self) -> Generator[Parameter, None, None]:
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, Module):
                yield from obj.parameters()
            elif isinstance(obj, Parameter):
                yield obj
