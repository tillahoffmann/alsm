import numpy as np


def evaluate_kernel(x: np.ndarray, y: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    """
    Evaluate the connectivity kernel.

    Args:
        x: Position of the first node with shape `(..., num_dims)`.
        y: Position of the second node with shape `(..., num_dims)`.
        propensity: Propensity for nodes to connect with shape `(...)`.

    Returns:
        Connectivity kernel with shape `(...)`.
    """
    delta2 = np.sum((x - y) ** 2, axis=-1)
    return propensity ** 2 * np.exp(- delta2 / 2)
