from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


def jacobian_loss(
    y_true: NDArray[np.number], y_pred: NDArray[np.number]
) -> NDArray[np.float32]:
    assert (
        y_true.shape == y_pred.shape
    ), f"Shapes of y_true and y_pred must be equal (Got: {y_true.shape} and {y_pred.shape})"

    return (y_pred - y_true).astype(np.float32)


# print(
#     jacobian_loss(
#         np.array([[1, 0], [0, 1], [1, 10]]), np.array([[0.9, 0.1], [0.2, 0.8], [2, 12]])
#     )
# )  # Expected output: array([0.05, 0.05], dtype=float32)
