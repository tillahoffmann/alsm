import cmdstanpy
import hashlib
import matplotlib as mpl
import matplotlib.collections
from matplotlib import pyplot as plt
import numpy as np
import pathlib
from scipy.linalg import orthogonal_procrustes
from scipy import special
from typing import Optional


def plot_edges(
    locs: np.ndarray,
    adjacency: np.ndarray,
    *,
    alpha_min: float = 0,
    alpha_max: float = 1,
    ax: mpl.axes.Axes = None,
    **kwargs,
) -> matplotlib.collections.LineCollection:
    # Validate the input.
    num_nodes, num_dims = locs.shape
    assert num_dims == 2, f"can only plot edges in two dimensions, not {num_dims}"
    assert adjacency.shape == (
        num_nodes,
        num_nodes,
    ), "adjacency matrix shape does not match the number of nodes"
    assert (
        0 <= alpha_min < alpha_max
    ), "alpha_min must be greater than or equal to zero and smaller than alpha_max"
    assert alpha_max <= 1, "alpha_max must be smaller than or equal to one"

    # Get edge indices and drop diagonals.
    edges = np.transpose(np.nonzero(adjacency))
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Construct the segments.
    segments = [[locs[i], locs[j]] for i, j in edges]

    # Set defaults.
    kwargs.setdefault("color", "k")
    if "alpha" not in kwargs:
        weights = adjacency[tuple(edges.T)]
        alpha_range = alpha_max - alpha_min
        kwargs["alpha"] = np.clip(
            alpha_min + alpha_range * weights / weights.max(), 0, 1
        )

    lines = mpl.collections.LineCollection(segments, **kwargs)
    (ax or plt.gca()).add_collection(lines)
    return lines


def align_samples(
    samples: np.ndarray, *extra_samples, reference: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Align samples of locations using rigid Procrustes transformations.

    Args:
        samples: Samples of locations with shape `(num_samples, num_units, num_dims)`.
        *extra_samples: Samples of locations with shape `(num_samples, *, num_dims)` to
            co-align with `samples`, i.e., we apply the same transform to them.
        reference: Reference to align with.

    Returns:
        aligned: Aligned samples.
    """
    # Center all the samples and the reference.
    samples = samples - samples.mean(axis=1, keepdims=True)
    extra_samples = [x - x.mean(axis=1, keepdims=True) for x in extra_samples]
    if reference is not None:
        reference = reference - reference.mean(axis=-2, keepdims=True)

    # Actual reference which we're going to align with.
    _reference = reference
    transformed = []
    extra_transformed = []
    for i, sample in enumerate(samples):
        sample = sample - sample.mean(axis=0)
        transform = np.eye(samples.shape[-1])
        # Automatically evaluate a reference if it wasn't given.
        if transformed and reference is None:
            _reference = np.mean(transformed, axis=0)
        if _reference is not None:
            transform, _ = orthogonal_procrustes(sample, _reference)
        transformed.append(sample @ transform)
        extra_transformed.append([x[i] @ transform for x in extra_samples])

    if extra_samples:
        return tuple(map(np.asarray, [transformed, *zip(*extra_transformed)]))
    return np.asarray(transformed)


def evaluate_grouping_matrix(
    group_idx: np.ndarray, num_groups: int = None, dtype=int
) -> np.ndarray:
    """
    Evaluate a matrix with shape `(num_groups, num_nodes)` that can be used to aggregate
    adjacency matrices.

    Args:
        group_idx: Group label for each node.
        num_groups: Number of groups (inferred from `group_idx` if not given).

    Returns:
        grouping: A grouping matrix.
    """
    (num_nodes,) = group_idx.shape
    num_groups = num_groups or group_idx.max() + 1
    np.testing.assert_array_less(group_idx, num_groups)
    grouping = np.zeros((num_groups, num_nodes), dtype)
    grouping[group_idx, np.arange(num_nodes)] = 1
    return grouping


def write_stanfile(code: str, directory: str = None) -> pathlib.Path:
    """
    Write the model code to a file and return the path based on the code hash.

    Args:
        code: Code to write to a file.
        directory: Directory to write the code to.

    Returns:
        path: Path to the file containing the code.
    """
    # Generate the path based on the code and check whether it already exists.
    directory = pathlib.Path(
        directory or (pathlib.Path(cmdstanpy.cmdstan_path()) / "models")
    )
    digest = hashlib.sha256(code.encode()).hexdigest()
    stan_file = directory / f"{digest}.stan"
    if stan_file.is_file():
        return stan_file

    # Write the file to disk.
    directory.mkdir(exist_ok=True)
    with open(stan_file, "w") as fp:
        fp.write(code)
    return stan_file


def get_samples(
    fit: cmdstanpy.CmdStanMCMC, param: str = None, flatten_chains: bool = False
) -> np.ndarray:
    """
    Get samples from a stan fit.

    Args:
        fit: Stan fit object to get samples from.
        param: Name of a parameter or `None` to get all parameters.
        flatten_chains: Whether to combine samples from all chains.

    Returns:
        samples: Posterior samples with shape `(*param_dims, num_samples * num_chains)`
            if `flatten_chains` is truthy and shape
            `(*param_dims, num_samples, num_chains)` otherwise.
    """
    samples = {}
    # Deal with the parameter samples (which come flattened by default).
    for key, value in fit.stan_variables().items():
        # Roll the sample axis to the back and reshape if we don't want to flatten.
        value = np.rollaxis(value, 0, value.ndim)
        if not flatten_chains:
            value = value.reshape(
                value.shape[:-1] + (fit.chains, fit.num_draws_sampling)
            )
            value = np.rollaxis(value, -2, value.ndim)
        samples[key] = value
    # Deal with the sampler-related parameters (which come unflattened by default).
    for key, value in fit.method_variables().items():
        if flatten_chains:
            value = value.reshape(fit.chains * fit.num_draws_sampling)
        samples[key] = value

    # Verify everything has the right shape.
    for value in samples.values():
        if flatten_chains:
            assert value.shape[-1] == fit.chains * fit.num_draws_sampling
        else:
            assert value.shape[-2:] == (fit.num_draws_sampling, fit.chains)

    if param:
        return samples[param]

    return samples


def get_chain(fit: cmdstanpy.CmdStanMCMC, chain) -> dict:
    """
    Get a particular chain from a stan fit.

    Args:
        fit: Stan fit object to get samples from.
        chain: Index of the chain to get or `best` to get the chain with the highest
            median `lp__`.

    Returns:
        chain: Dictionary mapping keys to samples of a particular chain.
    """
    if chain == "best":
        chain = np.median(fit.method_variables()["lp__"], axis=0).argmax()
    return {key: value[..., chain] for key, value in get_samples(fit).items()}


def evaluate_rotation_matrix(radians: float) -> np.ndarray:
    """
    Evaluate a two-dimensional rotation matrix.

    Args:
        radians: Angle of the rotation in radians.

    Returns:
        rotation: A rotation matrix.
    """
    return np.asarray(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )


def evaluate_elppd(log_likelihood, axis=-1, return_pwaic=False):
    """
    Evaluate the expected log posterior predictive density as discussed in BDA3.
    """
    lppd = special.logsumexp(log_likelihood, axis=axis).sum()
    pwaic2 = np.var(log_likelihood, axis=axis).sum()
    elppd = lppd - pwaic2
    if return_pwaic:
        return elppd, pwaic2
    return elppd


def estimate_mode(x: np.ndarray, scale: float = 3) -> np.ndarray:
    """
    Estimate the mode of a point cloud.

    Args:
        x: Point cloud with shape `(..., num_points, num_dims)`.
        scale: Scale factor for the kernel.

    Returns:
        center: Estimated centres with shape `(..., num_dims)`.
    """
    (
        *batch_shape,
        num_points,
        num_dims,
    ) = x.shape
    batch_shape = tuple(batch_shape)
    # Evaluate the distance between points.
    d2 = np.sum((x[..., :, None, :] - x[..., None, :, :]) ** 2, axis=-1)
    i = np.arange(num_points)
    d2[..., i, i] = np.inf
    assert d2.shape == batch_shape + (num_points, num_points)
    # Estimate the typical scale.
    s2 = np.median(d2.min(axis=-1), axis=-1) * scale**2
    assert s2.shape == batch_shape, (s2.shape, batch_shape)
    # Evaluate a Gaussian kernel and find the most central point by summing the kernel.
    d2[..., i, i] = 0
    score = np.sum(np.exp(-d2 / s2[..., None, None]), axis=-1)
    assert score.shape == batch_shape + (num_points,)
    # Index the input array to get the center.
    indices = np.indices(batch_shape)
    center = x[(*indices, score.argmax(axis=-1))]
    assert center.shape == batch_shape + (num_dims,)
    return center


def invert_index(index: np.ndarray) -> np.ndarray:
    """
    Inverted an index `index` such that `x[index][inverted] == x`.

    Args:
        index: Index to invert.

    Returns:
        inverted: Inverted index.
    """
    inverted = np.empty_like(index)
    inverted[index] = np.arange(index.size)
    return inverted


def get_elbo(approx: cmdstanpy.CmdStanVB) -> float:
    """
    Get the evidence lower bound from the log file of a variational fit.

    Args:
        approx: Variational approximation to the posterior of a model.

    Returns:
        elbo: Evidence lower bound if the fit has converged.
    """
    (filename,) = approx.runset.stdout_files
    with open(filename) as fp:
        elbo = None
        for line in fp:
            line = line.strip()
            if line.endswith("MEDIAN ELBO CONVERGED"):
                parts = line.split()
                assert len(parts) == 7
                elbo = float(parts[1])

    assert elbo is not None, "could not parse ELBO"
    return elbo
