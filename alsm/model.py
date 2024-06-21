import numpy as np
import re
from scipy import special, stats
import typing
from .util import evaluate_grouping_matrix


EPSILON: float = 1e-15

STAN_SNIPPETS = {
    # Evaluate a group scale given the fractional mean parameter.
    "evaluate_group_scale": """
        real evaluate_group_scale(real eta, int num_dims) {
            return sqrt((eta ^ (- 2.0 / num_dims) - 1) / 2);
        }
    """,
    # Evaluate the log Jacobian associated with the group scale transformation.
    "evaluate_group_scale_log_jac": """
        real evaluate_group_scale_log_jac(real eta, int num_dims) {
            return - log(eta ^ (- 2.0 / num_dims) - 1) / 2 - (2.0 + num_dims)
                / num_dims * log(eta);
        }
    """,
}


def stan_snippet(func: typing.Callable) -> typing.Callable:
    """
    Decorator to extract Stan code from an RST code block in the docstring and add it
    to the snippet library.
    """
    name = func.__name__
    if name in STAN_SNIPPETS:
        raise ValueError(f"{name} is already a snippet")  # pragma: no cover

    # Get the code from the docstring and ensure there's exactly one code block (cf
    # https://stackoverflow.com/a/53085201/1150961).
    pattern = r"\.\. code-block:: stan\s+$((\n +.*|\s)+)"
    matches = re.findall(pattern, func.__doc__, re.M)
    if (n := len(matches)) != 1:
        raise ValueError(
            f"need exactly one match for {name}, not {n}"
        )  # pragma: no cover

    # Add the code to the snippets and return the original function unmodified.
    code, _ = matches[0]
    STAN_SNIPPETS[name] = code
    func.__stan__ = code
    return func


@stan_snippet
def evaluate_kernel(x: np.ndarray, y: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the connectivity kernel :math:`\alpha \exp\left(-\frac{(x-y)^2}{2}\right)`.

    Args:
        x: Position of the first node with shape `(..., num_dims)`.
        y: Position of the second node with shape `(..., num_dims)`.
        propensity: Propensity for nodes to connect with shape `(...)`.

    Returns:
        kernel: Connectivity kernel with shape `(...)`.

    .. code-block:: stan

        real evaluate_kernel(vector x, vector y, real propensity) {
            real d2 = squared_distance(x, y);
            return propensity * exp(- d2 / 2);
        }
    """
    assert x.shape[-1] == y.shape[-1], "last dimension must match"
    delta2 = np.sum((x - y) ** 2, axis=-1)
    return propensity * np.exp(-delta2 / 2)


@stan_snippet
def evaluate_mean(
    x: np.ndarray,
    y: np.ndarray,
    xscale: np.ndarray,
    yscale: np.ndarray,
    propensity: np.ndarray,
) -> np.ndarray:
    r"""
    Evaluate the expected connectivity kernel :math:`\lambda_{ij}: for the members
    :math:`i` and :math:`j` of two clusters.

    .. code-block:: stan

        real evaluate_mean(vector x, vector y, real xscale, real yscale,
                           real propensity) {
            real d2 = squared_distance(x, y);
            real var_ = 1 + xscale ^ 2 + yscale ^ 2;
            int ndims = num_elements(x);
            return propensity * exp(- d2 / (2 * var_)) / var_ ^ (ndims / 2.0);
        }
    """
    d2 = np.square(x - y).sum(axis=-1)
    var = 1 + xscale**2 + yscale**2
    p = x.shape[-1]
    return propensity * np.exp(-d2 / (2 * var)) / var ** (p / 2)


@stan_snippet
def evaluate_square(
    x: np.ndarray,
    y: np.ndarray,
    xscale: np.ndarray,
    yscale: np.ndarray,
    propensity: np.ndarray,
) -> np.ndarray:
    r"""
    Evaluate the expected squared connectivity kernel :math:`\lambda_{ij}^2: for the
    members :math:`i` and :math:`j` of two clusters.

    .. code-block:: stan

        real evaluate_square(vector x, vector y, real xscale, real yscale,
                             real propensity) {
            real d2 = squared_distance(x, y);
            real var_ = 1 + 2 * (xscale ^ 2 + yscale ^ 2);
            int ndims = num_elements(x);
            return propensity ^ 2 * exp(- d2 / var_) / var_ ^ (ndims / 2.0);
        }
    """
    d2 = np.square(x - y).sum(axis=-1)
    var = 1 + 2 * (xscale**2 + yscale**2)
    p = x.shape[-1]
    return propensity**2 * np.exp(-d2 / var) / var ** (p / 2)


@stan_snippet
def evaluate_cross(
    x: np.ndarray,
    y: np.ndarray,
    xscale: np.ndarray,
    yscale: np.ndarray,
    propensity: np.ndarray,
) -> np.ndarray:
    r"""
    Evaluate the expected cross term :math:`\lambda_{ij}\lambda_{il}` fors members
    :math:`i`, :math:`j`, and :math:`l`, where :math:`i` belongs to the first cluster
    and :math:`j` and :math:`l` belong to the second cluster.

    .. code-block:: stan

        real evaluate_cross(vector x, vector y, real xscale, real yscale,
                            real propensity) {
            real d2 = squared_distance(x, y);
            real var_ = 1 + 2 * xscale ^ 2 + yscale ^ 2;
            int ndims = num_elements(x);
            return propensity ^ 2 * exp(- d2 / var_) / (var_ * (1 + yscale ^ 2))
                ^ (ndims / 2.0);
        }
    """
    d2 = np.square(x - y).sum(axis=-1)
    var = 1 + 2 * xscale**2 + yscale**2
    p = x.shape[-1]
    return propensity**2 * np.exp(-d2 / var) / (var * (1 + yscale**2)) ** (p / 2)


def evaluate_triplet(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xscale: np.ndarray,
    yscale: np.ndarray,
    zscale: np.ndarray,
    propensity: np.ndarray,
) -> np.ndarray:
    r"""
    Evaluate the expected triplet term :math:`\lambda_{ij}\lambda_{il}` fors members
    :math:`i`, :math:`j`, and :math:`l`, where :math:`i` belongs to the first cluster,
    and :math:`j` to the second, and :math:`l` to the third.
    """
    # Evaluate some reusable constants.
    p = x.shape[-1]
    xscale2 = xscale**2
    yscale2 = yscale**2
    zscale2 = zscale**2
    yscale21p = 1 + yscale2
    zscale21p = 1 + zscale2

    var_x = yscale21p + zscale21p
    loc_x = (y * zscale21p + z * yscale21p) / var_x
    norm = yscale21p * zscale21p + xscale2 * var_x
    chi2 = (
        var_x / norm * np.square(x - loc_x).sum(axis=-1)
        + np.square(y - z).sum(axis=-1) / var_x
    )
    return propensity**2 * np.exp(-chi2 / 2) / norm ** (p / 2)


@stan_snippet
def evaluate_aggregate_mean(
    x: np.ndarray,
    y: np.ndarray,
    xscale: np.ndarray,
    yscale: np.ndarray,
    propensity: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the expected connection volume :math:`Y_{ab}` between two clusters
    :math:`a` and :math:`b`.

    .. code-block:: stan

        real evaluate_aggregate_mean(vector x, vector y, real xscale, real yscale,
                                     real propensity, real n1, real n2) {
            real mean = evaluate_mean(x, y, xscale, yscale, propensity);
            if (n2 > 0) {
                return n1 * n2 * mean;
            } else {
                return n1 * (n1 - 1) * mean;
            }
        }
    """
    mean = evaluate_mean(x, y, xscale, yscale, propensity)
    if ny is None:
        return mean * nx * (nx - 1)
    else:
        return mean * nx * ny


@stan_snippet
def evaluate_aggregate_var(
    x: np.ndarray,
    y: np.ndarray,
    xscale: np.ndarray,
    yscale: np.ndarray,
    propensity: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    weighted: bool,
) -> np.ndarray:
    """
    Evaluate the variance of the connection volume :math:`Y_{ab}` between two clusters
    :math:`a` and :math:`b`.

    .. code-block:: stan

        real evaluate_aggregate_var(vector x, vector y, real xscale, real yscale,
                                    real propensity, real n1, real n2, int weighted) {
            real y_ij = evaluate_mean(x, y, xscale, yscale, propensity);
            real y_ijkl = y_ij ^ 2;
            real y_ijji = evaluate_square(x, y, xscale, yscale, propensity);
            real y_ijij = y_ij + (weighted ? y_ijji : 0);
            real y_ijil = evaluate_cross(x, y, xscale, yscale, propensity);
            real y_ijkj = evaluate_cross(y, x, yscale, xscale, propensity);

            // Between group connections.
            if (n2 > 0) {
                return n1 * n2 * (
                    y_ijij + (n2 - 1) * y_ijil + (n1 - 1) * y_ijkj
                    - (n1 + n2 - 1) * y_ijkl
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
    y_ijkl = y_ij**2
    y_ijji = evaluate_square(x, y, xscale, yscale, propensity)
    y_ijij = y_ij + (y_ijji if weighted else 0)
    y_ijil = evaluate_cross(x, y, xscale, yscale, propensity)
    y_ijkj = evaluate_cross(y, x, yscale, xscale, propensity)

    if ny is None:
        return (
            nx
            * (nx - 1)
            * (y_ijij + y_ijji + 4 * (nx - 2) * y_ijil - 2 * (2 * nx - 3) * y_ijkl)
        )
    else:
        return (
            nx
            * ny
            * (y_ijij + (ny - 1) * y_ijil + (nx - 1) * y_ijkj - (nx + ny - 1) * y_ijkl)
        )


def evaluate_aggregate_cov(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xscale: np.ndarray,
    yscale: np.ndarray,
    zscale: np.ndarray,
    propensity: np.ndarray,
    n1: int,
    n2: int,
    n3: int,
) -> np.ndarray:
    """
    Evaluate the covariance between inter-group aggregate connection volumes
    :math:`Y_{ab}` and :math:`Y_{ac}`.

    Args:
        x:
        y:
        z:
        xscale:
        yscale:
        zscale:
        propensity:
        n1:
        n2:
        n3:

    Returns:
        cov:
    """
    if z is None:
        assert (
            zscale is None and n3 is None
        ), "all or none of `z`, `zscale`, and `n3` must be given"
        triplet = evaluate_triplet(x, x, y, xscale, xscale, yscale, propensity)
        mean_prod = evaluate_mean(x, x, xscale, xscale, propensity) * evaluate_mean(
            x, y, xscale, yscale, propensity
        )
        return 2 * n1 * (n1 - 1) * n2 * (triplet - mean_prod)
    else:
        triplet = evaluate_triplet(x, y, z, xscale, yscale, zscale, propensity)
        mean_prod = evaluate_mean(x, y, xscale, yscale, propensity) * evaluate_mean(
            x, z, xscale, zscale, propensity
        )
        return n1 * n2 * n3 * (triplet - mean_prod)


def evaluate_kernel_pdf(
    kernel: np.ndarray,
    delta: np.ndarray,
    variance: np.ndarray,
    num_dims: np.ndarray,
    propensity: np.ndarray,
) -> np.ndarray:
    r"""
    Evaluate the likelihood of observing a particular kernel value given the separation
    between two clusters, the sum of their variances, and the number of dimensions.

    Args:
        kernel: Value at which to evaluate the density function.
        delta: Distance between the two clusters.
        variance: Sum of variances of the two clusters (or twice the variance for
            within-cluster connections).
        num_dims: Number of dimensions of the latent space.

    Returns:
        pdf: Probability density function evaluated at the kernel value given other
            parameters.

    Note:
        The sum of squares of the distance between two nodes follows a scaled
        non-central :math:`\chi^2` distribution, i.e.

        .. math::
            s^2=\sum_{i=1}^q (z_i - z_j)^2\sim\sigma^2
            \chi_q\left(\frac{\delta^2}{\sigma^2}\right).

        We can then obtain the distribution of kernel values by noting that
        :math:`\lambda=\exp\left(-\frac{s^2}{2}\right)` and computing the Jacobian.
    """
    kernel = kernel / propensity
    bessel_arg = np.sqrt(-2 * np.log(kernel) * (delta / variance) ** 2)
    log = (
        (1 / variance - 1) * np.log(kernel)
        - np.log(variance)
        - delta**2 / (2 * variance)
        + (num_dims / 4 - 0.5) * np.log(-2 * np.log(kernel) / delta**2)
        + np.log(special.ive(num_dims / 2 - 1, bessel_arg))
        + bessel_arg
    )
    return np.exp(log) / propensity


def evaluate_neg_binomial_np(
    mean: np.ndarray, var: np.ndarray, epsilon: float = EPSILON
) -> typing.Tuple[np.ndarray, np.ndarray]:
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
    n = mean**2 / excess_var
    p = mean / var
    return n, p


@stan_snippet
def evaluate_neg_binomial_2_phi(
    mean: np.ndarray, variance: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    """
    Evaluate the concentration parameter phi of the alternative parametrisation of the
    negative binomial distribution.

    .. code-block:: stan

        real evaluate_neg_binomial_2_phi(real mean, real variance, real epsilon) {
            real invphi = (variance - mean) / mean ^ 2;
            return 1 / fmax(invphi, epsilon);
        }
    """
    invphi = (variance - mean) / mean**2
    # If the inverse of the phi parameter in the negbinom_2 parametrisation is negative,
    # the model is underdispersed with respect to a Poisson distribution. This will
    # never be the case in our model, but we may very well end up with negative values
    # due to numerical issues. We'll just return a large, fixed phi.
    return 1 / np.maximum(invphi, epsilon)


@stan_snippet
def neg_binomial_mv_lpmf(
    x: np.ndarray, mean: np.ndarray, variance: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    """
    Evaluate the log probability mass function of the beta binomial distribution.

    .. code-block:: stan

        real neg_binomial_mv_lpmf(int x, real mean, real variance, real epsilon) {
            real phi = evaluate_neg_binomial_2_phi(mean, variance, epsilon);
            return neg_binomial_2_lpmf(x | mean, phi);
        }
    """
    n, p = evaluate_neg_binomial_np(mean, variance, epsilon)
    return stats.nbinom(n, p).logpmf(x)


@stan_snippet
def neg_binomial_mv_rng(
    mean: np.ndarray, variance: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    """
    Draw a sample from the beta binomial distribution.

    .. code-block:: stan

        int neg_binomial_mv_rng(real mean, real variance, real epsilon) {
            real phi = evaluate_neg_binomial_2_phi(mean, variance, epsilon);
            return neg_binomial_2_rng(fmax(mean, epsilon), phi);
        }
    """
    n, p = evaluate_neg_binomial_np(mean, variance, epsilon)
    return np.random.negative_binomial(n, p)


@stan_snippet
def evaluate_beta_binomial_phi(
    trials: np.ndarray, mean: np.ndarray, variance: np.ndarray, epsilon: float = 1e-9
) -> np.ndarray:
    """
    Convert mean, variance, and number of trials to the concentration parameter of a
    beta distribution.

    .. code-block:: stan

        real evaluate_beta_binomial_phi(real trials, real mean, real variance,
                                        real epsilon) {
            real mu = fmax(mean / trials, epsilon);
            real ratio = variance / (trials * mu * (1 - mu));
            real phi = (trials - ratio) / fmax(ratio - 1, epsilon);
            if (phi <= 0 || is_nan(phi)) {
                reject("beta binomial phi=", phi, " is not positive; trials=",
                       trials, ", mean=", mean, ", variance=", variance);
            }
            return phi;
        }
    """
    # This ratio is between trials and trials ** 2, and we have
    # ratio == (trials + phi) / (1 + phi). The ratio is bounded below by one
    # (corresponding to a standard binomial distribution) and bounded above by the
    # number of trials (corresponding to a maximal overdispersion).
    mu = mean / trials
    ratio = variance / (trials * mu * (1 - mu))
    phi = (trials - ratio) / np.maximum(ratio - 1, epsilon)
    np.testing.assert_array_less(
        0,
        phi,
        f"beta binomial phi={phi} is not positive; "
        f"trials={trials}, mean={mean}, variance={variance}",
    )
    return phi


def evaluate_beta_binomial_ab(
    trials: np.ndarray, mean: np.ndarray, var: np.ndarray, epsilon: float = 1e-9
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Convert mean, variance, and number of trials to the shape parameters of a beta
    binomial distribution.

    Args:
        trials: Number of trials.
        mean: Mean of the distribution.
        var: Variance of the distribution.

    Returns:
        a: First shape parameter of the success probability beta distribution.
        b: Second shape parameter of the success probability beta distribution.
    """
    phi = evaluate_beta_binomial_phi(trials, mean, var, epsilon)
    mu = np.maximum(mean / trials, epsilon)
    return mu * phi, (1 - mu) * phi


@stan_snippet
def beta_binomial_mv_lpmf(
    x: np.ndarray,
    trials: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    Evaluate the log probability mass function of the beta binomial distribution.

    .. code-block:: stan

        real beta_binomial_mv_lpmf(int x, int trials, real mean, real variance,
                                   real epsilon) {
            real conc = evaluate_beta_binomial_phi(trials, mean, variance, epsilon);
            real loc = fmax(mean / trials, epsilon);
            return beta_binomial_lpmf(x | trials, conc * loc, conc * (1 - loc));
        }
    """
    a, b = evaluate_beta_binomial_ab(trials, mean, variance, epsilon)
    return stats.betabinom(trials, a, b).logpmf(x)


@stan_snippet
def beta_binomial_mv_rng(
    trials: np.ndarray, mean: np.ndarray, variance: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    """
    Draw a sample from the beta binomial distribution.

    .. code-block:: stan

        int beta_binomial_mv_rng(int trials, real mean, real variance, real epsilon) {
            real conc = evaluate_beta_binomial_phi(trials, mean, variance, epsilon);
            real loc = fmax(mean / trials, epsilon);
            return beta_binomial_rng(trials, conc * loc, conc * (1 - loc));
        }
    """
    a, b = evaluate_beta_binomial_ab(trials, mean, variance, epsilon)
    return stats.betabinom(trials, a, b).rvs()


def get_group_model_code() -> str:
    """
    Get the Stan code for the group-level model.
    """
    return """
    functions {
        %(all_snippets)s
    }

    data {
        int<lower=0, upper=1> weighted;
        int<lower=1> num_groups;
        int<lower=1> num_dims;
        array [num_groups, num_groups] int group_adjacency;
        array [num_groups] int group_sizes;
        real<lower=0> epsilon;
    }

    transformed data {
        // Evaluate the number of trials for connections between different groups.
        array [num_groups, num_groups] int<lower=0> num_trials;
        for (i in 1:num_groups) {
            num_trials[i, i] = group_sizes[i] * (group_sizes[i] - 1);
            for (j in i + 1:num_groups) {
                num_trials[i, j] = group_sizes[i] * group_sizes[j];
                num_trials[j, i] = num_trials[i, j];
            }
        }
    }

    // Parameters of the model.
    parameters {
        real<lower=0> population_scale;
        vector[num_dims] center;
        cholesky_factor_cov[num_groups - 1, num_dims] group_locs_raw_;
        real<lower=0, upper=1> propensity;
        // This is the fraction of the potential mean within-group connections we can
        // have.
        vector<lower=0, upper=1>[num_groups] eta;
    }

    // Estimate moments of the aggregate relational data.
    transformed parameters {
        array [num_groups] vector[num_dims] group_locs;
        vector<lower=0>[num_groups] group_scales;
        array [num_groups, num_groups] real mu;
        array [num_groups, num_groups] real variance;

        // Evaluate the group locations.
        group_locs[1] = center;
        for (i in 2:num_groups) {
            group_locs[i] = center + group_locs_raw_[i - 1]';
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
                        group_locs[i], group_locs[j], group_scales[i], group_scales[j],
                        propensity, group_sizes[i], 0, weighted
                    );
                }
                // Evaluate between-group connections.
                else {
                    mu[i, j] = mu[i, j] * group_sizes[i] * group_sizes[j];
                    variance[i, j] = evaluate_aggregate_var(
                        group_locs[i], group_locs[j], group_scales[i], group_scales[j],
                        propensity, group_sizes[i], group_sizes[j], weighted
                    );
                }
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
                if (weighted) {
                    group_adjacency[i, j] ~ neg_binomial_mv(mu[i, j], variance[i, j],
                                                            epsilon);
                } else {
                    group_adjacency[i, j] ~ beta_binomial_mv(num_trials[i, j], mu[i, j],
                                                             variance[i, j], epsilon);
                }
            }
        }
    }

    // Generate posterior predictive replicates.
    generated quantities {
        array [num_groups, num_groups] int ppd_group_adjacency;
        for (i in 1:num_groups) {
            for (j in 1:num_groups) {
                if (weighted) {
                    ppd_group_adjacency[i, j] = neg_binomial_mv_rng(
                        mu[i, j], variance[i, j], epsilon);
                } else {
                    ppd_group_adjacency[i, j] = beta_binomial_mv_rng(
                        num_trials[i, j], mu[i, j], variance[i, j], epsilon);
                }
            }
        }
    }
""" % {
        "all_snippets": "\n".join(STAN_SNIPPETS.values())
    }


def get_individual_model_code(group_prior: bool) -> str:
    """
    Get the Stan code for the individual-level model.

    Args:
        group_prior: Whether to use a Gaussian group-level prior or a global Cauchy
            prior.
    """
    lines = {}

    lines["data"] = [
        "int<lower=1> num_nodes;",
        "int<lower=1> num_dims;",
        "array [num_nodes, num_nodes] int<lower=0, upper=1> adjacency;",
        "real<lower=0> epsilon;",
    ]

    lines["transformed data"] = [
        "array [num_nodes * num_nodes] int<lower=0, upper=1> flat_adjacency;",
        "for (i in 1:num_nodes) { for (j in 1: num_nodes) { "
        "flat_adjacency[(i - 1) * num_nodes + j] = adjacency[i, j]; } }",
    ]

    lines["parameters"] = [
        "real<lower=0, upper=1> propensity;",
        "array [num_nodes] vector[num_dims] locs;",
    ]

    lines["transformed parameters"] = []

    lines["model"] = [
        # We can evaluate the kernel efficiently using the Gaussian process covariance
        # matrix. We evaluate the kernel in the model section so we don't have excessive
        # IO.
        "matrix[num_nodes, num_nodes] kernel = fmax("
        "gp_exp_quad_cov(locs, sqrt(propensity), 1), epsilon);",
        "for (i in 1:num_nodes) { kernel[i, i] = 0; }",
        "propensity ~ beta(1, 1);",
        "flat_adjacency ~ bernoulli(to_vector(kernel));",
    ]

    if group_prior:
        lines["data"].extend(
            [
                "int<lower=1> num_groups;",
                "array [num_nodes] int<lower=1, upper=num_groups> group_idx;",
            ]
        )
        lines["parameters"].extend(
            [
                "array [num_groups] vector[num_dims] group_locs;",
                "array [num_groups] real<lower=0> group_scales;",
                "real<lower=0> population_scale;",
            ]
        )
        lines["model"].extend(
            [
                "group_scales ~ cauchy(0, 1);",
                "for (i in 1:num_groups) "
                "{ group_locs[i] ~ normal(0, population_scale); }",
                "for (i in 1:num_nodes) { int group = group_idx[i]; locs[i] "
                "~ normal(group_locs[group], group_scales[group]); }",
            ]
        )
    else:
        lines["data"].extend(
            [
                "real<lower=0> population_scale;",
            ]
        )
        lines["model"].extend(
            [
                "for (i in 1:num_nodes) { locs[i] ~ normal(0, population_scale); }",
            ]
        )

    lines = {key: "\n".join(value) for key, value in lines.items()}
    return "\n".join(f"{key} {{{value}}}" for key, value in lines.items())


def _generate_common_data(
    group_sizes: np.ndarray, num_dims: int, weighted, **params
) -> dict:
    params["group_sizes"] = group_sizes
    params["num_dims"] = num_dims
    params["weighted"] = 1 if weighted else 0
    (num_groups,) = (params["num_groups"],) = group_sizes.shape
    params["num_nodes"] = group_sizes.sum()

    # Sample the overall propensity to form edges.
    params.setdefault(
        "propensity",
        np.random.uniform(0, 1),
    )

    # Sample the scales and positions of nodes.
    population_scale = params.setdefault(
        "population_scale",
        np.abs(np.random.standard_cauchy()),
    )
    params.setdefault(
        "group_locs",
        np.random.normal(0, population_scale, (num_groups, num_dims)),
    )
    params.setdefault(
        "group_scales",
        np.abs(np.random.standard_cauchy(num_groups)),
    )

    # Evaluate the number of trials.
    trials = group_sizes[:, None] * group_sizes[None, :]
    np.fill_diagonal(trials, group_sizes * (group_sizes - 1))
    params.setdefault("group_trials", trials)
    return params


def generate_data(
    group_sizes: np.ndarray, num_dims: int, weighted: bool, **params
) -> dict:
    """
    Generate data from a latent space model with groups.

    Args:
        group_sizes: Number of nodes belonging to each group.
        num_dims: Number of dimensions of the latent space.

    Returns:
        data: Synthetic dataset.
    """
    params = _generate_common_data(group_sizes, num_dims, weighted, **params)

    group_idx = params["group_idx"] = np.repeat(
        np.arange(params["num_groups"]), group_sizes
    )
    assert group_idx.shape == (params["num_nodes"],)
    locs = params.setdefault(
        "locs",
        np.random.normal(
            params["group_locs"][group_idx], params["group_scales"][group_idx, None]
        ),
    )

    # Evaluate the kernel, removing self-edges.
    kernel = params["kernel"] = evaluate_kernel(
        locs[:, None, :], locs[None, :, :], params["propensity"]
    )
    assert kernel.shape == (params["num_nodes"], params["num_nodes"])
    np.fill_diagonal(kernel, 0)

    # Sample the adjacency matrix and aggregate it.
    adjacency = params.setdefault(
        "adjacency",
        np.random.poisson(kernel) if weighted else np.random.binomial(1, kernel),
    )
    grouping = evaluate_grouping_matrix(group_idx, params["num_groups"])
    params["group_adjacency"] = (grouping @ adjacency @ grouping.T).astype(int)
    return params


def generate_group_data(
    group_sizes: np.ndarray, num_dims: int, weighted: bool, **params
) -> dict:
    params = _generate_common_data(group_sizes, num_dims, weighted, **params)
    group_locs = params["group_locs"]
    group_scales = params["group_scales"]

    # Evaluate the mean and variance for between-group connections.
    inter_args = (
        group_locs[:, None],
        group_locs[None, :],
        group_scales[:, None],
        group_scales[None, :],
        params["propensity"],
        group_sizes[:, None],
        group_sizes[None, :],
    )
    mean = evaluate_aggregate_mean(*inter_args)
    var = evaluate_aggregate_var(*inter_args, weighted=weighted)
    assert mean.shape == params["group_trials"].shape
    assert var.shape == params["group_trials"].shape

    # Evaluate the diagonal terms.
    intra_args = (
        group_locs,
        group_locs,
        group_scales,
        group_scales,
        params["propensity"],
        group_sizes,
        None,
    )
    np.fill_diagonal(mean, evaluate_aggregate_mean(*intra_args))
    np.fill_diagonal(var, evaluate_aggregate_var(*intra_args, weighted=weighted))

    # Sample the grouped adjacency matrix.
    if weighted:
        group_adjacency = neg_binomial_mv_rng(mean, var)
    else:
        group_adjacency = beta_binomial_mv_rng(params["group_trials"], mean, var)
    params.setdefault("group_adjacency", group_adjacency)
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
    # Verify that the index truely is a permutation.
    np.testing.assert_array_equal(np.sort(index), np.arange(index.size))
    y = {}
    for key, value in x.items():
        if key in {"group_locs", "group_scales", "group_sizes", "eta"}:
            value = value[index]
        elif key in {"group_adjacency", "ppd_group_adjacency", "mu", "variance", "phi"}:
            value = value[index][:, index]
        y[key] = value
    return y
