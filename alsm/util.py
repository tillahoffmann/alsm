import matplotlib as mpl
import matplotlib.collections
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import orthogonal_procrustes


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


def evaluate_grouping_matrix(group_idx: np.ndarray, num_groups: int = None) -> np.ndarray:
    """
    Evaluate a matrix with shape `(num_groups, num_nodes)` that can be used to aggregate adjacency
    matrices.

    Args:
        group_idx: Group label for each node.
        num_groups: Number of groups (inferred from `group_idx` if not given).

    Returns:
        A grouping matrix.
    """
    num_nodes, = group_idx.shape
    num_groups = num_groups or group_idx.max() + 1
    np.testing.assert_array_less(group_idx, num_groups)
    grouping = np.zeros((num_groups, num_nodes))
    grouping[group_idx, np.arange(num_nodes)] = 1
    return grouping


def generate_data(group_sizes: np.ndarray, num_dims: int, **params) -> dict:
    """
    Generate data from a latent space model with groups.

    Args:
        group_sizes: Number of nodes belonging to each group.
        num_dims: Number of dimensions of the latent space.

    Returns:
        Synthetic dataset.
    """
    params['group_sizes'] = group_sizes
    params['num_dims'] = num_dims
    num_groups, = params['num_groups'], = group_sizes.shape
    num_nodes = params['num_nodes'] = group_sizes.sum()

    # Sample the scales and positions of nodes.
    population_scale = params.setdefault(
        'population_scale',
        np.abs(np.random.standard_cauchy()),
    )
    group_locs = params.setdefault(
        'group_locs',
        np.random.normal(0, population_scale, (num_groups, num_dims)),
    )
    group_scales = params.setdefault(
        'group_scales',
        np.abs(np.random.standard_cauchy(num_groups)),
    )
    group_idx = params['group_idx'] = np.repeat(np.arange(num_groups), group_sizes)
    assert group_idx.shape == (num_nodes,)
    locs = params.setdefault(
        'locs',
        np.random.normal(group_locs[group_idx], group_scales[group_idx, None])
    )

    # Sample the propensity and evaluate the kernel, removing self-edges.
    propensity = params.setdefault(
        'propensity',
        np.random.uniform(0, 1),
    )
    kernel = params['kernel'] = evaluate_kernel(locs[:, None, :], locs[None, :, :], propensity)
    assert kernel.shape == (num_nodes, num_nodes)
    np.fill_diagonal(kernel, 0)

    # Sample the adjacency matrix and aggregate it.
    adjacency = params.setdefault(
        'adjacency',
        np.random.poisson(kernel),
    )
    grouping = evaluate_grouping_matrix(group_idx, num_groups)
    params['group_adjacency'] = (grouping @ adjacency @ grouping.T).astype(int)
    return params


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
        Aligned samples.
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
