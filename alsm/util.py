import matplotlib as mpl
import matplotlib.collections
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import orthogonal_procrustes
import stan.fit
import typing


def plot_edges(locs: np.ndarray, adjacency: np.ndarray, *, alpha_min: float = 0,
               alpha_max: float = 1, ax: mpl.axes.Axes = None, **kwargs) \
                   -> matplotlib.collections.LineCollection:
    # Validate the input.
    num_nodes, num_dims = locs.shape
    assert num_dims == 2, f'can only plot edges in two dimensions, not {num_dims}'
    assert adjacency.shape == (num_nodes, num_nodes), \
        'adjacency matrix shape does not match the number of nodes'
    assert 0 <= alpha_min < alpha_max, \
        'alpha_min must be greater than or equal to zero and smaller than alpha_max'
    assert alpha_max <= 1, 'alpha_max must be smaller than or equal to one'

    # Get edge indices and drop diagonals.
    edges = np.transpose(np.nonzero(adjacency))
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Construct the segments.
    segments = [[locs[i], locs[j]] for i, j in edges]

    # Set defaults.
    kwargs.setdefault('color', 'k')
    if 'alpha' not in kwargs:
        weights = adjacency[tuple(edges.T)]
        alpha_range = alpha_max - alpha_min
        kwargs['alpha'] = np.clip(alpha_min + alpha_range * weights / weights.max(), 0, 1)

    lines = mpl.collections.LineCollection(segments, **kwargs)
    (ax or plt.gca()).add_collection(lines)
    return lines


def align_samples(samples: np.ndarray) -> np.ndarray:
    """
    Align samples of locations using rigid Procrustes transformations.

    Args:
        samples: Samples of locations with shape `(num_samples, num_units, num_dims)`.

    Returns:
        aligned: Aligned samples.
    """
    transformed = []
    for sample in samples:
        sample = sample - sample.mean(axis=0)
        if transformed:
            reference = np.mean(transformed, axis=0)
            transform, _ = orthogonal_procrustes(sample, reference)
            sample = sample @ transform
        transformed.append(sample)
    return np.asarray(transformed)


def evaluate_grouping_matrix(group_idx: np.ndarray, num_groups: int = None) -> np.ndarray:
    """
    Evaluate a matrix with shape `(num_groups, num_nodes)` that can be used to aggregate adjacency
    matrices.

    Args:
        group_idx: Group label for each node.
        num_groups: Number of groups (inferred from `group_idx` if not given).

    Returns:
        grouping: A grouping matrix.
    """
    num_nodes, = group_idx.shape
    num_groups = num_groups or group_idx.max() + 1
    np.testing.assert_array_less(group_idx, num_groups)
    grouping = np.zeros((num_groups, num_nodes))
    grouping[group_idx, np.arange(num_nodes)] = 1
    return grouping


def get_samples(fit: stan.fit.Fit, param: str, flatten_chains: bool = True, squeeze: bool = True) \
        -> np.ndarray:
    """
    Get samples from a stan fit.

    Args:
        fit: Stan fit object to get samples from.
        param: Name of the parameter.
        flatten_chains: Whether to combine samples from all chains.
        squeeze: Whether to remove dimensions of unit size.

    Returns:
        samples: Posterior samples with shape `(*param_dims, num_samples * num_chains)` if
            `flatten_chains` is truthy and shape `(*param_dims, num_samples, num_chains)` otherwise.
    """
    samples = fit[param]
    if not flatten_chains:
        *shape, _ = samples.shape
        shape = shape + [-1, fit.num_chains]
        samples = samples.reshape(shape)
    if squeeze:
        samples = np.squeeze(samples)
    return samples


def get_chain(fit: stan.fit.Fit, chain, squeeze=True) -> dict:
    """
    Get a particular chain from a stan fit.

    Args:
        fit: Stan fit object to get samples from.
        chain: Index of the chain to get or `best` to get the chain with the highest median `lp__`.
        squeeze: Whether to remove dimensions of unit size.

    Returns:
        chain: Dictionary mapping keys to samples of a particular chain.
    """
    if chain == 'best':
        chain = np.median(get_samples(fit, 'lp__', False), axis=0).argmax()
    return {
        key: get_samples(fit, key, False, squeeze)[..., chain]
        for key in fit.sample_and_sampler_param_names + fit.param_names
    }


def negative_binomial_np(mean: np.ndarray, var: np.ndarray, epsilon: float = 1e-9) \
        -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Convert mean and variance to the parameters of a negative binomial distribution.

    Args:
        mean: Mean of the distribution.
        var: Variance of the distribution.

    Returns:
        n: Number of failures for the negative binomial distribution.
        p: Success probability of each trial.
    """
    excess_var = np.maximum(var - mean, epsilon)
    n = mean ** 2 / excess_var
    p = mean / var
    return n, p
