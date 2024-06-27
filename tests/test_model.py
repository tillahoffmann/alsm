from alsm import util as alsm_util
from alsm import model as alsm_model
import cmdstanpy
import matplotlib.axes
import numpy as np
import pytest
from scipy import integrate, stats
import typing


# Generate samples that we're going to use to compare with theoretical summary
# statistics.
NUM_DIMS = 2
NUM_SAMPLES = 1000
NUM_BOOTSTRAP = 999
PROPENSITY = np.random.uniform(0, 1)
N1 = 12
N2 = 15
N3 = 17
GROUP_SCALE1, GROUP_SCALE2, GROUP_SCALE3 = np.random.gamma(10, 0.1, 3)
GROUP_LOC1, GROUP_LOC2, GROUP_LOC3 = np.random.normal(0, 1, (3, NUM_DIMS))
X = np.random.normal(0, 1, (NUM_SAMPLES, N1, NUM_DIMS)) * GROUP_SCALE1 + GROUP_LOC1
Y, Yp = (
    np.random.normal(0, 1, (2, NUM_SAMPLES, N2, NUM_DIMS)) * GROUP_SCALE2 + GROUP_LOC2
)
Z = np.random.normal(0, 1, (NUM_SAMPLES, N3, NUM_DIMS)) * GROUP_SCALE3 + GROUP_LOC3

# Evaluate summary statistics reused across tests.
KERNEL_XY = alsm_model.evaluate_kernel(X[:, 0], Y[:, 0], PROPENSITY)
KERNEL_XYp = alsm_model.evaluate_kernel(X[:, 0], Yp[:, 0], PROPENSITY)
KERNEL_XZ = alsm_model.evaluate_kernel(X[:, 0], Z[:, 0], PROPENSITY)

# Evaluate ARD samples for within- and between-group connections.
KERNEL_INTRA = alsm_model.evaluate_kernel(
    X[:, :, None, :], X[:, None, :, :], PROPENSITY
)
i = np.arange(N1)
KERNEL_INTRA[:, i, i] = 0
KERNEL_INTER = alsm_model.evaluate_kernel(
    X[:, :, None, :], Y[:, None, :, :], PROPENSITY
)
KERNEL_INTER_XZ = alsm_model.evaluate_kernel(
    X[:, :, None, :], Z[:, None, :, :], PROPENSITY
)


@pytest.fixture(params=[True, False], ids=["weighted", "unweighted"])
def weighted(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def ard_intra(weighted: bool) -> np.ndarray:
    if weighted:
        x = np.random.poisson(KERNEL_INTRA)
    else:
        x = np.random.binomial(1, KERNEL_INTRA)
    return x.sum(axis=(1, 2))


@pytest.fixture
def ard_inter(weighted: bool) -> np.ndarray:
    if weighted:
        x = np.random.poisson(KERNEL_INTER)
    else:
        x = np.random.binomial(1, KERNEL_INTER)
    return x.sum(axis=(1, 2))


@pytest.fixture
def ard_inter_xz(weighted: bool) -> np.ndarray:
    if weighted:
        x = np.random.poisson(KERNEL_INTER_XZ)
    else:
        x = np.random.binomial(1, KERNEL_INTER_XZ)
    return x.sum(axis=(1, 2))


def _bootstrap(
    xs: np.ndarray, x: float, func=np.mean, ax: matplotlib.axes.Axes = None
) -> None:
    """
    Bootstrap the mean of `xs` and compare with the theoretical value `x`.
    """
    num_samples = xs.shape[0]
    # Evaluate the bootstrap samples.
    bootstrap = np.asarray(
        [
            func(xs[np.random.randint(num_samples, size=num_samples)], axis=0)
            for _ in range(NUM_BOOTSTRAP)
        ]
    )

    # Evaluate the pvalue and z-score.
    pvalue = (x < bootstrap).mean()
    pvalue = min(pvalue, 1 - pvalue)
    z = (x - func(xs, axis=0)) / bootstrap.std()

    if ax:
        ax.hist(bootstrap, density=True, bins=20)
        ax.axvline(x, color="k", ls=":")
        lines = [
            f"p-value: {pvalue:.5f}",
            f"z-score: {z:.5f}",
        ]
        bbox = {
            "facecolor": "w",
            "alpha": 0.5,
        }
        ax.text(
            0.05, 0.95, "\n".join(lines), va="top", transform=ax.transAxes, bbox=bbox
        )

    # Ensure at least one sample lies on either side of the theoretical value.
    assert pvalue > 1 / NUM_BOOTSTRAP, f"p-value: {pvalue}; z-score: {z}"
    assert np.abs(z) < 10, f"p-value: {pvalue}; z-score: {z}"


def _stan_python_identity(
    func: typing.Callable,
    return_type: str,
    args: list,
    helper_functions: list = None,
    use_bar=False,
) -> typing.Callable:
    # Evaluate the python version of the function.
    python_data = {key: value for _, key, value in args if not key.endswith("_")}
    python = func(**python_data)

    # Evaluate the stan version of the function.
    def _stan_lookup(x):
        try:
            return {None: 0, True: 1, False: 0}[x]
        except (KeyError, TypeError):
            return x

    data = {key: _stan_lookup(value) for _, key, value in args}
    stan_data = "\n".join([f"{type} {name};" for type, name, _ in args])
    stan_code = [x.__stan__ for x in helper_functions or []] + [func.__stan__]
    arg_names = [name for _, name, _ in args if not name.endswith("_")]
    if use_bar:
        arg_names = "| ".join([arg_names[0], ", ".join(arg_names[1:])])
    else:
        arg_names = ", ".join(arg_names)
    posterior = cmdstanpy.CmdStanModel(
        stan_file=alsm_util.write_stanfile(
            """
    functions {
        %(stan_code)s
    }

    data {
        %(stan_data)s
    }

    transformed parameters {
        %(return_type)s value = %(func_name)s(%(arg_names)s);
    }
    """
            % {
                "stan_data": stan_data,
                "stan_code": "\n".join(stan_code),
                "func_name": func.__name__,
                "arg_names": arg_names,
                "return_type": return_type,
            }
        )
    )
    # We need to increase the number of significant figures for the comparison to work
    # reliably (cf https://bit.ly/3oFp0y7).
    fit = posterior.sample(
        data=data, chains=1, iter_sampling=1, fixed_param=True, sig_figs=15
    )
    np.testing.assert_allclose(python, fit.stan_variable("value"))
    return python


def test_evaluate_kernel():
    _stan_python_identity(
        alsm_model.evaluate_kernel,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", np.random.normal(0, 1, NUM_DIMS)),
            ("vector[k_]", "y", np.random.normal(0, 1, NUM_DIMS)),
            ("real", "propensity", np.random.uniform(0, 1)),
        ],
    )


def test_evaluate_mean():
    mean = _stan_python_identity(
        alsm_model.evaluate_mean,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
        ],
    )
    _bootstrap(KERNEL_XY, mean)


def test_evaluate_log_mean():
    _stan_python_identity(
        alsm_model.evaluate_log_mean,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
        ],
    )


def test_evaluate_square():
    square = _stan_python_identity(
        alsm_model.evaluate_square,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
        ],
    )
    _bootstrap(KERNEL_XY**2, square)


def test_evaluate_log_square():
    _stan_python_identity(
        alsm_model.evaluate_log_square,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
        ],
    )


def test_evaluate_cross():
    cross = _stan_python_identity(
        alsm_model.evaluate_cross,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
        ],
    )
    _bootstrap(KERNEL_XY * KERNEL_XYp, cross)


def test_evaluate_log_cross():
    _stan_python_identity(
        alsm_model.evaluate_log_cross,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
        ],
    )


def test_evaluate_log_cov():
    _stan_python_identity(
        alsm_model.evaluate_log_cov,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
        ],
    )


def test_evaluate_triplet():
    triplet = alsm_model.evaluate_triplet(
        GROUP_LOC1,
        GROUP_LOC2,
        GROUP_LOC3,
        GROUP_SCALE1,
        GROUP_SCALE2,
        GROUP_SCALE3,
        PROPENSITY,
    )
    _bootstrap(KERNEL_XY * KERNEL_XZ, triplet)


def test_evaluate_aggregate_mean_intra(ard_intra: np.ndarray):
    aggregate_mean_intra = _stan_python_identity(
        alsm_model.evaluate_aggregate_mean,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC1),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE1),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", None),
        ],
        [alsm_model.evaluate_mean],
    )
    _bootstrap(ard_intra, aggregate_mean_intra)


def test_evaluate_aggregate_mean_inter(ard_inter: np.ndarray):
    aggregate_mean_inter = _stan_python_identity(
        alsm_model.evaluate_aggregate_mean,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", N2),
        ],
        [alsm_model.evaluate_mean],
    )
    _bootstrap(ard_inter, aggregate_mean_inter)


def test_evaluate_log_aggregate_mean_intra():
    _stan_python_identity(
        alsm_model.evaluate_log_aggregate_mean,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC1),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE1),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", None),
        ],
        [alsm_model.evaluate_log_mean],
    )


def test_evaluate_log_aggregate_mean_inter():
    _stan_python_identity(
        alsm_model.evaluate_log_aggregate_mean,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", N2),
        ],
        [alsm_model.evaluate_log_mean],
    )


def test_evaluate_aggregate_var_intra(ard_intra: np.ndarray, weighted: bool):
    aggregate_var_intra = _stan_python_identity(
        alsm_model.evaluate_aggregate_var,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC1),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE1),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", None),
            ("int<lower=0, upper=1>", "weighted", weighted),
        ],
        [
            alsm_model.evaluate_mean,
            alsm_model.evaluate_square,
            alsm_model.evaluate_cross,
        ],
    )
    _bootstrap(ard_intra, aggregate_var_intra, func=np.var)


def test_evaluate_aggregate_var_inter(ard_inter: np.ndarray, weighted: bool):
    aggregate_var_inter = _stan_python_identity(
        alsm_model.evaluate_aggregate_var,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", N2),
            ("int<lower=0, upper=1>", "weighted", weighted),
        ],
        [
            alsm_model.evaluate_mean,
            alsm_model.evaluate_square,
            alsm_model.evaluate_cross,
        ],
    )
    _bootstrap(ard_inter, aggregate_var_inter, func=np.var)


def test_evaluate_log_aggregate_var_intra(weighted: bool):
    _stan_python_identity(
        alsm_model.evaluate_log_aggregate_var,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC1),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE1),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", None),
            ("int<lower=0, upper=1>", "weighted", weighted),
        ],
        [
            alsm_model.evaluate_log_mean,
            alsm_model.evaluate_log_square,
            alsm_model.evaluate_log_cross,
        ],
    )


def test_evaluate_log_aggregate_var_inter(weighted: bool):
    _stan_python_identity(
        alsm_model.evaluate_log_aggregate_var,
        "real",
        [
            ("int", "k_", NUM_DIMS),
            ("vector[k_]", "x", GROUP_LOC1),
            ("vector[k_]", "y", GROUP_LOC2),
            ("real<lower=0>", "xscale", GROUP_SCALE1),
            ("real<lower=0>", "yscale", GROUP_SCALE2),
            ("real<lower=0, upper=1>", "propensity", PROPENSITY),
            ("int<lower=0>", "nx", N1),
            ("int<lower=0>", "ny", N2),
            ("int<lower=0, upper=1>", "weighted", weighted),
        ],
        [
            alsm_model.evaluate_log_mean,
            alsm_model.evaluate_log_square,
            alsm_model.evaluate_log_cross,
        ],
    )


def test_evaluate_aggregate_cov_intra_inter(
    ard_intra: np.ndarray, ard_inter: np.ndarray
):
    cov = alsm_model.evaluate_aggregate_cov(
        GROUP_LOC1,
        GROUP_LOC2,
        None,
        GROUP_SCALE1,
        GROUP_SCALE2,
        None,
        PROPENSITY,
        N1,
        N2,
        None,
    )
    _bootstrap(
        np.transpose((ard_intra, ard_inter)), cov, lambda xs, axis: np.cov(*xs.T)[0, 1]
    )


def test_evaluate_aggregate_cov_inter_inter(
    ard_inter: np.ndarray, ard_inter_xz: np.ndarray
):
    cov = alsm_model.evaluate_aggregate_cov(
        GROUP_LOC1,
        GROUP_LOC2,
        GROUP_LOC3,
        GROUP_SCALE1,
        GROUP_SCALE2,
        GROUP_SCALE3,
        PROPENSITY,
        N1,
        N2,
        N3,
    )
    _bootstrap(
        np.transpose((ard_inter, ard_inter_xz)),
        cov,
        lambda xs, axis: np.cov(*xs.T)[0, 1],
    )


def test_evaluate_beta_binomial_phi():
    phi = _stan_python_identity(
        alsm_model.evaluate_beta_binomial_phi,
        "real",
        [
            ("int", "trials", 20),
            ("real", "mean", 10),
            ("real", "variance", 7),
            ("real", "epsilon", 1e-9),
        ],
    )
    assert phi > 0


def test_evaluate_beta_binomial_log_phi():
    _stan_python_identity(
        alsm_model.evaluate_beta_binomial_log_phi,
        "real",
        [
            ("int", "trials", 20),
            ("real", "log_mean", 2),
            ("real", "log_variance", 3),
        ],
    )


def test_evaluate_neg_binomial_2_phi():
    phi = _stan_python_identity(
        alsm_model.evaluate_neg_binomial_2_phi,
        "real",
        [
            ("real", "mean", 10),
            ("real", "variance", 12),
            ("real", "epsilon", 1e-9),
        ],
    )
    assert phi > 0


def test_evaluate_neg_binomial_2_log_inv_phi():
    _stan_python_identity(
        alsm_model.evaluate_neg_binomial_2_log_inv_phi,
        "real",
        [
            ("real", "log_mean", 3),
            ("real", "log_variance", 3.2),
        ],
    )


def test_beta_binom_mv_lpmf():
    trials = 100
    dist = stats.betabinom(trials, 3, 7)
    x = dist.rvs()
    _stan_python_identity(
        alsm_model.beta_binomial_mv_lpmf,
        "real",
        [
            ("int", "x", x),
            ("int", "trials", trials),
            ("real", "mean", dist.mean()),
            ("real", "variance", dist.var()),
            ("real", "epsilon", 1e-9),
        ],
        [alsm_model.evaluate_beta_binomial_phi],
        use_bar=True,
    )


def test_beta_binom_lmv_lpmf():
    trials = 100
    dist = stats.betabinom(trials, 3, 7)
    x = dist.rvs()
    lpmf = _stan_python_identity(
        alsm_model.beta_binomial_lmv_lpmf,
        "real",
        [
            ("int", "x", x),
            ("int", "trials", trials),
            ("real", "log_mean", np.log(dist.mean())),
            ("real", "log_variance", np.log(dist.var())),
            ("real", "epsilon", 1e-9),
        ],
        [alsm_model.evaluate_beta_binomial_phi],
        use_bar=True,
    )
    np.testing.assert_allclose(lpmf, dist.logpmf(x))


def test_neg_binom_mv_lpmf():
    dist = stats.nbinom(10, 0.2)
    x = dist.rvs()
    _stan_python_identity(
        alsm_model.neg_binomial_mv_lpmf,
        "real",
        [
            ("int", "x", x),
            ("real", "mean", dist.mean()),
            ("real", "variance", dist.var()),
            ("real", "epsilon", 1e-9),
        ],
        [alsm_model.evaluate_neg_binomial_2_phi],
        use_bar=True,
    )


def test_neg_binom_lmv_lpmf():
    dist = stats.nbinom(10, 0.2)
    x = dist.rvs()
    lpmf = _stan_python_identity(
        alsm_model.neg_binomial_lmv_lpmf,
        "real",
        [
            ("int", "x", x),
            ("real", "log_mean", np.log(dist.mean())),
            ("real", "log_variance", np.log(dist.var())),
            ("real", "epsilon", 1e-9),
        ],
        [alsm_model.evaluate_neg_binomial_2_phi],
        use_bar=True,
    )
    np.testing.assert_allclose(lpmf, dist.logpmf(x))


@pytest.mark.parametrize(
    "scale_prior_type", ["normal", "cauchy", "jeffrey", "exponential"]
)
@pytest.mark.parametrize("group_data", [False, True])
def test_group_model(group_data: bool, weighted: bool, scale_prior_type: str):
    num_dims = 4
    generator = (
        alsm_model.generate_group_data if group_data else alsm_model.generate_data
    )
    data = generator(
        np.asarray([10, 20, 30, 40, 50]), num_dims, weighted, population_scale=1
    )
    data["epsilon"] = 1e-6
    model_code = alsm_model.get_group_model_code(scale_prior_type=scale_prior_type)
    stan_file = alsm_util.write_stanfile(model_code)
    posterior = cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = posterior.sample(data=data, chains=4, iter_sampling=5, iter_warmup=17)
    assert fit.chains == 4


def test_group_scale_change_of_variables(figure):
    data = {"num_dims": 2, "alpha": 5, "beta": 2}
    posterior = cmdstanpy.CmdStanModel(
        stan_file=alsm_util.write_stanfile(
            """
    functions {
        %(evaluate_group_scale)s
        %(evaluate_group_scale_log_jac)s
    }

    data {
        int<lower=1> num_dims;
        real<lower=0> alpha, beta;
    }

    parameters {
        real<lower=0, upper=1> eta;
    }

    transformed parameters {
        real<lower=0> group_scale = evaluate_group_scale(eta, num_dims);
    }

    model {
        group_scale ~ gamma(alpha, beta);
        target += evaluate_group_scale_log_jac(eta, num_dims);
    }
    """
            % alsm_model.STAN_SNIPPETS
        )
    )
    fit = posterior.sample(iter_sampling=1000, data=data)

    # Get the samples and thin them because the samples may still have autocorrelation
    # (which will mess with the bootstrap estimate).
    xs = alsm_util.get_samples(fit, "group_scale", flatten_chains=False)
    assert xs.shape == (1000, 4)
    xs = xs[::10].ravel()

    gs = figure.add_gridspec(2, 2)

    ax = figure.add_subplot(gs[:, 0])
    ax.hist(xs, bins=20, density=True)
    lin = np.linspace(0, xs.max(), 100)
    ax.plot(
        lin, stats.gamma(data["alpha"], scale=1 / data["beta"]).pdf(lin), color="C1"
    )
    ax.set_xlabel(r"$\eta\sim\mathrm{Gamma}(%(alpha).1f, %(beta).1f)$" % data)

    ax = figure.add_subplot(gs[0, 1])
    ax.set_xlabel("bootstrapped mean")
    _bootstrap(xs, data["alpha"] / data["beta"], ax=ax)

    ax = figure.add_subplot(gs[1, 1])
    ax.set_xlabel("bootstrapped var")
    _bootstrap(xs, data["alpha"] / data["beta"] ** 2, func=np.var, ax=ax)


def test_generate_data(weighted: bool):
    group_sizes = np.asarray([10, 20, 30])
    (num_groups,) = group_sizes.shape
    num_nodes = group_sizes.sum()
    num_dims = 2
    data = alsm_model.generate_data(group_sizes, num_dims, weighted)
    assert data["population_scale"].shape == ()
    assert data["adjacency"].shape == (num_nodes, num_nodes)
    assert data["group_adjacency"].shape == (num_groups, num_groups)
    assert data["adjacency"].sum() == data["group_adjacency"].sum()
    assert data["group_locs"].shape == (num_groups, num_dims)
    assert data["group_scales"].shape == (num_groups,)
    assert data["locs"].shape == (num_nodes, num_dims)


def test_apply_permutation_index(weighted: bool):
    data = alsm_model.generate_data(np.asarray([10, 20, 30]), 2, weighted)
    index = np.random.permutation(3)
    permuted = alsm_model.apply_permutation_index(data, index)
    actual = alsm_model.apply_permutation_index(permuted, alsm_util.invert_index(index))
    for key, value in data.items():
        np.testing.assert_array_equal(value, actual[key])


def test_negbinom_mean_var_to_params():
    mean = np.random.gamma(1, 1)
    var = mean + np.random.gamma(1, 1)

    dist = stats.nbinom(*alsm_model.evaluate_neg_binomial_np(mean, var))
    np.testing.assert_allclose(mean, dist.mean())
    np.testing.assert_allclose(var, dist.var())


def test_betabinom_mean_var_to_params():
    trials = 100
    a, b = np.random.gamma(10, size=2)
    dist = stats.betabinom(trials, a, b)

    a_, b_ = alsm_model.evaluate_beta_binomial_ab(trials, dist.mean(), dist.var())
    np.testing.assert_allclose(a, a_)
    np.testing.assert_allclose(b, b_)


@pytest.mark.parametrize("group_prior", [False, True])
def test_individual_model(group_prior: bool):
    code = alsm_model.get_individual_model_code(group_prior)
    posterior = cmdstanpy.CmdStanModel(stan_file=alsm_util.write_stanfile(code))
    data = alsm_model.generate_data(
        np.asarray([10, 20, 30]), 2, False, population_scale=1
    )
    data.update(
        {
            "epsilon": alsm_model.EPSILON,
            "group_idx": data["group_idx"] + 1,
        }
    )
    fit = posterior.sample(data, iter_warmup=1, iter_sampling=1, chains=1)
    np.testing.assert_array_less(fit.log_likelihood, 1e-9)


def test_evaluate_kernel_pdf():
    PROPENSITY = 1
    delta = np.linalg.norm(GROUP_LOC1 - GROUP_LOC2)
    variance = GROUP_SCALE1**2 + GROUP_SCALE2**2
    mean = alsm_model.evaluate_mean(
        GROUP_LOC1, GROUP_LOC2, GROUP_SCALE1, GROUP_SCALE2, PROPENSITY
    )
    y, _ = integrate.quad(
        lambda x: x
        * alsm_model.evaluate_kernel_pdf(x, delta, variance, NUM_DIMS, PROPENSITY),
        1e-9,
        PROPENSITY - 1e-9,
    )
    np.testing.assert_allclose(mean, y)


def test_neg_binomial_lmv_rng():
    mean = 10
    var = 11

    np.random.seed(0)
    a = alsm_model.neg_binomial_mv_rng(mean, var)
    np.random.seed(0)
    b = alsm_model.neg_binomial_lmv_rng(np.log(mean), np.log(var))
    np.testing.assert_allclose(a, b)


def test_beta_binomial_lmv_rng():
    trials = 100
    mean = 10
    var = 10

    np.random.seed(0)
    a = alsm_model.beta_binomial_mv_rng(trials, mean, var)
    np.random.seed(0)
    b = alsm_model.beta_binomial_lmv_rng(trials, np.log(mean), np.log(var))
    np.testing.assert_allclose(a, b)
