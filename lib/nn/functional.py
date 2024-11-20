from typing import Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.number)


def relu(x: NDArray[T]) -> NDArray[T]:
    return np.maximum(x, 0)


def sigmoid(x: NDArray[np.number]):
    """
    Args:
        x (NDArray[T]): (num_classes, num_samples) matrix

    Returns:
        NDArray[T]:
    """
    return 1 / (1 + np.exp(-x))


def one_hot_encode(
    y: NDArray[np.integer], num_classes: Optional[int] = None
) -> NDArray[np.integer]:
    """
    Args:
        y (NDArray[np.integer]): list of labels

    Returns:
        NDArray[np.integer]: <num_classes> x <num_samples> one-hot encoded labels
    """
    num_classes = num_classes if num_classes is not None else np.max(y) + 1
    one_hot_Y = np.zeros((y.size, num_classes), dtype=np.int32)  # type: ignore
    one_hot_Y[np.arange(y.size), y] = 1
    return one_hot_Y.T
