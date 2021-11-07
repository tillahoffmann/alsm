import numpy as np
import re
import typing
from .util import evaluate_grouping_matrix, negative_binomial_np


STAN_SNIPPETS = {
    # Evaluate the concentration parameter phi of the alternative parametrisation of the negative
    # binomial distribution.
    'evaluate_negbinom_2_phi': """
        real evaluate_negbinom_2_phi(real mean, real var_, real epsilon) {
            real invphi = (var_ - mean) / mean ^ 2;
            // If the inverse of the phi parameter in the negbinom_2 parametrisation is negative,
            // the model is underdispersed with respect to a Poisson distribution. This will never
            // be the case in our model, but we may very well end up with negative values due to
            // numerical issues. We'll just return a large, fixed phi.
            return 1 / fmax(invphi, epsilon);
        }
    """,
    # Evaluate a group scale given the fractional mean parameter.
    'evaluate_group_scale': """
        real evaluate_group_scale(real eta, int num_dims) {
            return sqrt((eta ^ (- 2.0 / num_dims) - 1) / 2);
        }
    """,
    # Evaluate the log Jacobian associated with the group scale transformation.
    'evaluate_group_scale_log_jac': """
        real evaluate_group_scale_log_jac(real eta, int num_dims) {
            return - log(eta ^ (- 2.0 / num_dims) - 1) / 2 - (2.0 + num_dims) / num_dims * log(eta);
        }
    """
}


def stan_snippet(func: typing.Callable) -> typing.Callable:
    """
    Decorator to extract Stan code from an RST code block in the docstring and add it to the snippet
    library.
    """
    name = func.__name__
    if name in STAN_SNIPPETS:
        raise ValueError(f'{name} is already a snippet')  # pragma: no cover

    # Get the code from the docstring and ensure there's exactly one code block (cf
    # https://stackoverflow.com/a/53085201/1150961).
    pattern = r"\.\. code-block:: stan\s+$((\n +.*|\s)+)"
    matches = re.findall(pattern, func.__doc__, re.M)
    if (n := len(matches)) != 1:
        raise ValueError(f'need exactly one match for {name}, not {n}')  # pragma: no cover

    # Add the code to the snippets and return the original function unmodified.
    code, _ = matches[0]
    STAN_SNIPPETS[name] = code
    return func


def evaluate_kernel(x: np.ndarray, y: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the connectivity kernel :math:`\alpha \exp\left(-\frac{(x-y)^2}{2}\right)`.

    Args:
        x: Position of the first node with shape `(..., num_dims)`.
        y: Position of the second node with shape `(..., num_dims)`.
        propensity: Propensity for nodes to connect with shape `(...)`.

    Returns:
        kernel: Connectivity kernel with shape `(...)`.
    """
    delta2 = np.sum((x - y) ** 2, axis=-1)
    return propensity * np.exp(- delta2 / 2)


@stan_snippet
def evaluate_mean(x: np.ndarray, y: np.ndarray, xscale: np.ndarray, yscale: np.ndarray,
                  propensity: np.ndarray) -> np.ndarray:

    r"""
    Evaluate the expected connectivity kernel :math:`\lambda_{ij}: for the members :math:`i` and
    :math:`j` of two clusters.

    .. code-block:: stan

        real evaluate_mean(vector x, vector y, real xscale, real yscale, real propensity) {
            real d2 = squared_distance(x, y);
            real var_ = 1 + xscale ^ 2 + yscale ^ 2;
            int ndims = num_elements(x);
            return propensity * exp(- d2 / (2 * var_)) / var_ ^ (ndims / 2.0);
        }
    """
    d2 = np.square(x - y).sum(axis=-1)
    var = 1 + xscale ** 2 + yscale ** 2
    p = x.shape[-1]
    return propensity * np.exp(- d2 / (2 * var)) / var ** (p / 2)


@stan_snippet
def evaluate_square(x: np.ndarray, y: np.ndarray, xscale: np.ndarray, yscale: np.ndarray,
                    propensity: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the expected squared connectivity kernel :math:`\lambda_{ij}^2: for the members
    :math:`i` and :math:`j` of two clusters.

    .. code-block:: stan

        real evaluate_square(vector x, vector y, real xscale, real yscale, real propensity) {
            real d2 = squared_distance(x, y);
            real var_ = 1 + 2 * (xscale ^ 2 + yscale ^ 2);
            int ndims = num_elements(x);
            return propensity ^ 2 * exp(- d2 / var_) / var_ ^ (ndims / 2.0);
        }
    """
    d2 = np.square(x - y).sum(axis=-1)
    var = 1 + 2 * (xscale ** 2 + yscale ** 2)
    p = x.shape[-1]
    return propensity ** 2 * np.exp(- d2 / var) / var ** (p / 2)


@stan_snippet
def evaluate_cross(x: np.ndarray, y: np.ndarray, xscale: np.ndarray, yscale: np.ndarray,
                   propensity: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the expected cross term :math:`\lambda_{ij}\lambda_{il}` fors members :math:`i`,
    :math:`j`, and :math:`l`, where :math:`i` belongs to the first cluster and :math:`j` and
    :math:`l` belong to the second cluster.

    .. code-block:: stan

        real evaluate_cross(vector x, vector y, real xscale, real yscale, real propensity) {
            real d2 = squared_distance(x, y);
            real var_ = 1 + 2 * xscale ^ 2 + yscale ^ 2;
            int ndims = num_elements(x);
            return propensity ^ 2 * exp(- d2 / var_) / (var_ * (1 + yscale ^ 2)) ^ (ndims / 2.0);
        }
    """
    d2 = np.square(x - y).sum(axis=-1)
    var = 1 + 2 * xscale ** 2 + yscale ** 2
    p = x.shape[-1]
    return propensity ** 2 * np.exp(- d2 / var) / (var * (1 + yscale ** 2)) ** (p / 2)


@stan_snippet
def evaluate_aggregate_mean(x: np.ndarray, y: np.ndarray, xscale: np.ndarray, yscale: np.ndarray,
                            propensity: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """
    Evaluate the expected connection volume :math:`Y_{ab}` between two clusters :math:`a` and
    :math:`b`.

    .. code-block:: stan

        real evaluate_aggregate_mean(vector x, vector y, real xscale, real yscale, real propensity,
                                     real n1, real n2) {
            real mean_ = evaluate_mean(x, y, xscale, yscale, propensity);
            if (n2 > 0) {
                return n1 * n2 * mean_;
            } else {
                return n1 * (n1 - 1) * mean_;
            }
        }
    """
    mean = evaluate_mean(x, y, xscale, yscale, propensity)
    if ny is None:
        return mean * nx * (nx - 1)
    else:
        return mean * nx * ny


@stan_snippet
def evaluate_aggregate_var(x: np.ndarray, y: np.ndarray, xscale: np.ndarray, yscale: np.ndarray,
                           propensity: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """
    Evaluate the variance of the connection volume :math:`Y_{ab}` between two clusters :math:`a` and
    :math:`b`.

    .. code-block:: stan

        real evaluate_aggregate_var(vector x, vector y, real xscale, real yscale, real propensity,
                                    real n1, real n2) {
            real y_ij = evaluate_mean(x, y, xscale, yscale, propensity);
            real y_ijkl = y_ij ^ 2;
            real y_ijji = evaluate_square(x, y, xscale, yscale, propensity);
            real y_ijij = y_ij + y_ijji;
            real y_ijil = evaluate_cross(x, y, xscale, yscale, propensity);
            real y_ijkj = evaluate_cross(y, x, yscale, xscale, propensity);

            // Between group connections.
            if (n2 > 0) {
                return n1 * n2 * (
                    y_ijij + (n2 - 1) * y_ijil + (n1 - 1) * y_ijkj - (n1 + n2 - 1) * y_ijkl
                );
            }
            // Within group connections.
            else {
                return n1 * (n1 - 1) * (
                    y_ijij + y_ijji + 4 * (n1 - 2) * y_ijil - 2 * (2 * n1 - 3) * y_ijkl
                );
            }
        }
    """
    y_ij = evaluate_mean(x, y, xscale, yscale, propensity)
    y_ijkl = y_ij ** 2
    y_ijji = evaluate_square(x, y, xscale, yscale, propensity)
    y_ijij = y_ij + y_ijji
    y_ijil = evaluate_cross(x, y, xscale, yscale, propensity)
    y_ijkj = evaluate_cross(y, x, yscale, xscale, propensity)

    if ny is None:
        return nx * (nx - 1) * (y_ijij + y_ijji + 4 * (nx - 2) * y_ijil - 2 * (2 * nx - 3) * y_ijkl)
    else:
        return nx * ny * (y_ijij + (ny - 1) * y_ijil + (nx - 1) * y_ijkj - (nx + ny - 1) * y_ijkl)


def get_group_model_code() -> str:
    """
    Get the Stan code for the group-level model.
    """
    return """
    functions {
        %(all_snippets)s
    }

    data {
        int<lower=1> num_groups;
        int<lower=1> num_dims;
        int group_adjacency[num_groups, num_groups];
        vector[num_groups] group_sizes;
        real<lower=0> epsilon;
    }

    // Parameters of the model.
    parameters {
        real<lower=0> population_scale;
        vector[num_dims] center;
        cholesky_factor_cov[num_groups - 1, num_dims] _group_locs_raw;
        real<lower=0, upper=1> propensity;
        // This is the fraction of the potential mean within-group connections we can have.
        vector<lower=0, upper=1>[num_groups] eta;
    }

    // Estimate parameters of the negative binomial distribution.
    transformed parameters {
        vector[num_dims] group_locs[num_groups];
        vector<lower=0>[num_groups] group_scales;
        real mu[num_groups, num_groups];
        real variance[num_groups, num_groups];
        real phi[num_groups, num_groups];

        // Evaluate the group locations.
        group_locs[1] = center;
        for (i in 2:num_groups) {
            group_locs[i] = center + _group_locs_raw[i - 1]';
        }

        // Obtain the group scales based on the "fraction of the maximum possible mean".
        for (i in 1:num_groups) {
            group_scales[i] = evaluate_group_scale(eta[i], num_dims);
        }

        for (i in 1:num_groups) {
            for (j in 1:num_groups) {
                mu[i, j] = evaluate_mean(group_locs[i], group_locs[j], group_scales[i],
                                         group_scales[j], propensity);
                // Evaluate within-group connections.
                if (i == j) {
                    mu[i, j] = mu[i, j] * group_sizes[i] * (group_sizes[i] - 1);
                    variance[i, j] = evaluate_aggregate_var(
                        group_locs[i], group_locs[j], group_scales[i], group_scales[j], propensity,
                        group_sizes[i], 0
                    );
                }
                // Evaluate between-group connections.
                else {
                    mu[i, j] = mu[i, j] * group_sizes[i] * group_sizes[j];
                    variance[i, j] = evaluate_aggregate_var(
                        group_locs[i], group_locs[j], group_scales[i], group_scales[j], propensity,
                        group_sizes[i], group_sizes[j]
                    );
                }

                // Threshold the mean and variance such that we don't end up with numerical issues
                // when the connection rate is small.
                mu[i, j] = fmax(mu[i, j], epsilon);
                variance[i, j] = fmax(variance[i, j], epsilon);

                // Evaluate the overdispersion parameter for the negative binomial distribution.
                phi[i, j] = evaluate_negbinom_2_phi(mu[i, j], variance[i, j], epsilon);
            }
        }
    }

    // The actual model.
    model {
        propensity ~ beta(1, 1);
        group_scales ~ cauchy(0, 1);
        population_scale ~ cauchy(0, 1);

        for (i in 1:num_groups) {
            group_locs[i] ~ normal(0, population_scale);
            // Account for the change of variables.
            target += evaluate_group_scale_log_jac(eta[i], num_dims);
            for (j in 1:num_groups) {
                group_adjacency[i, j] ~ neg_binomial_2(mu[i, j], phi[i, j]);
            }
        }
    }

    // Generate posterior predictive replicates.
    generated quantities {
        int ppd_group_adjacency[num_groups, num_groups];
        for (i in 1:num_groups) {
            for (j in 1:num_groups) {
                ppd_group_adjacency[i, j] = neg_binomial_2_rng(mu[i, j], phi[i, j]);
            }
        }
    }
""" % {'all_snippets': '\n'.join(STAN_SNIPPETS.values())}


def _generate_common_data(group_sizes: np.ndarray, num_dims: int, **params) -> dict:
    params['group_sizes'] = group_sizes
    params['num_dims'] = num_dims
    num_groups, = params['num_groups'], = group_sizes.shape
    params['num_nodes'] = group_sizes.sum()

    # Sample the overall propensity to form edges.
    params.setdefault(
        'propensity',
        np.random.uniform(0, 1),
    )

    # Sample the scales and positions of nodes.
    population_scale = params.setdefault(
        'population_scale',
        np.abs(np.random.standard_cauchy()),
    )
    params.setdefault(
        'group_locs',
        np.random.normal(0, population_scale, (num_groups, num_dims)),
    )
    params.setdefault(
        'group_scales',
        np.abs(np.random.standard_cauchy(num_groups)),
    )
    return params


def generate_data(group_sizes: np.ndarray, num_dims: int, **params) -> dict:
    """
    Generate data from a latent space model with groups.

    Args:
        group_sizes: Number of nodes belonging to each group.
        num_dims: Number of dimensions of the latent space.

    Returns:
        data: Synthetic dataset.
    """
    params = _generate_common_data(group_sizes, num_dims, **params)

    group_idx = params['group_idx'] = np.repeat(np.arange(params['num_groups']), group_sizes)
    assert group_idx.shape == (params['num_nodes'],)
    locs = params.setdefault(
        'locs',
        np.random.normal(params['group_locs'][group_idx], params['group_scales'][group_idx, None])
    )

    # Evaluate the kernel, removing self-edges.
    kernel = params['kernel'] = evaluate_kernel(locs[:, None, :], locs[None, :, :],
                                                params['propensity'])
    assert kernel.shape == (params['num_nodes'], params['num_nodes'])
    np.fill_diagonal(kernel, 0)

    # Sample the adjacency matrix and aggregate it.
    adjacency = params.setdefault(
        'adjacency',
        np.random.poisson(kernel),
    )
    grouping = evaluate_grouping_matrix(group_idx, params['num_groups'])
    params['group_adjacency'] = (grouping @ adjacency @ grouping.T).astype(int)
    return params


def generate_group_data(group_sizes: np.ndarray, num_dims: int, **params) -> dict:
    params = _generate_common_data(group_sizes, num_dims, **params)
    group_locs = params['group_locs']
    group_scales = params['group_scales']

    # Evaluate the mean and variance for between-group connections.
    inter_args = (
        group_locs[:, None], group_locs[None, :], group_scales[:, None], group_scales[None, :],
        params['propensity'], group_sizes[:, None], group_sizes[None, :]
    )
    mean = evaluate_aggregate_mean(*inter_args)
    var = evaluate_aggregate_var(*inter_args)
    assert mean.shape == (params['num_groups'], params['num_groups'])
    assert var.shape == mean.shape

    # Evaluate the diagonal terms.
    intra_args = (
        group_locs, group_locs, group_scales, group_scales, params['propensity'], group_sizes, None,
    )
    np.fill_diagonal(mean, evaluate_aggregate_mean(*intra_args))
    np.fill_diagonal(var, evaluate_aggregate_var(*intra_args))

    # Sample the grouped adjacency matrix.
    n, p = negative_binomial_np(mean, var)
    params.setdefault('group_adjacency', np.random.negative_binomial(n, p))
    return params


def apply_permutation_index(x: dict, index: np.ndarray) -> dict:
    """
    Apply a permutation index to data or posterior samples.

    Args:
        x: Mapping to apply the permutation to.
        index: Permutation index to apply.

    Returns:
        y: Mapping after application of the permutation index.
    """
    y = {}
    for key, value in x.items():
        if key in {'group_locs', 'group_scales', 'group_sizes', 'eta'}:
            value = value[index]
        elif key in {'group_adjacency', 'ppd_group_adjacency', 'mu', 'variance', 'phi'}:
            value = value[index][:, index]
        y[key] = value
    return y
