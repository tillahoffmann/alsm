import alsm
import itertools as it
import matplotlib.axes
import numpy as np
import pytest
from scipy import stats
import stan
import stan.fit


# Generate samples that we're going to use to compare with theoretical summary statistics.
NUM_DIMS = 2
NUM_SAMPLES = 1000
NUM_BOOTSTRAP = 999
PROPENSITY = np.random.uniform(0, 1)
N1 = 12
N2 = 15
GROUP_SCALE1, GROUP_SCALE2 = np.random.gamma(10, .1, 2)
GROUP_LOC1, GROUP_LOC2 = np.random.normal(0, 1, (2, NUM_DIMS))
X = np.random.normal(0, 1, (NUM_SAMPLES, N1, NUM_DIMS)) * GROUP_SCALE1 + GROUP_LOC1
Y, Yp = np.random.normal(0, 1, (2, NUM_SAMPLES, N2, NUM_DIMS)) * GROUP_SCALE2 + GROUP_LOC2

# Evaluate summary statistics reused across tests.
KERNEL_XY = alsm.evaluate_kernel(X[:, 0], Y[:, 0], PROPENSITY)
KERNEL_XYp = alsm.evaluate_kernel(X[:, 0], Yp[:, 0], PROPENSITY)

# Evaluate ARD samples for within- and between-group connections.
KERNEL_INTRA = alsm.evaluate_kernel(X[:, :, None, :], X[:, None, :, :], PROPENSITY)
i = np.arange(N1)
KERNEL_INTRA[:, i, i] = 0
ARD_INTRA = np.random.poisson(KERNEL_INTRA).sum(axis=(1, 2))

KERNEL_INTER = alsm.evaluate_kernel(X[:, :, None, :], Y[:, None, :, :], PROPENSITY)
ARD_INTER = np.random.poisson(KERNEL_INTER).sum(axis=(1, 2))


def _bootstrap(xs: np.ndarray, x: float, func=np.mean, ax: matplotlib.axes.Axes = None):
    """
    Bootstrap the mean of `xs` and compare with the theoretical value `x`.
    """
    assert xs.ndim == 1, 'bootstrap tests only supported for scalar variables'
    num_samples, = xs.shape
    # Evaluate the bootstrap samples.
    bootstrap = np.asarray([
        func(xs[np.random.randint(num_samples, size=num_samples)], axis=0)
        for _ in range(NUM_BOOTSTRAP)
    ])

    # Evaluate the pvalue and z-score.
    pvalue = (x < bootstrap).mean()
    pvalue = min(pvalue, 1 - pvalue)
    z = (x - func(xs)) / bootstrap.std()

    if ax:
        ax.hist(bootstrap, density=True, bins=20)
        ax.axvline(x, color='k', ls=':')
        lines = [
            f'p-value: {pvalue:.5f}',
            f'z-score: {z:.5f}',
        ]
        bbox = {
            'facecolor': 'w',
            'alpha': 0.5,
        }
        ax.text(0.05, 0.95, '\n'.join(lines), va='top', transform=ax.transAxes, bbox=bbox)

    # Ensure at least one sample lies on either side of the theoretical value.
    assert pvalue > 1 / NUM_BOOTSTRAP, f'p-value: {pvalue}; z-score: {z}'
    assert np.abs(z) < 10, f'p-value: {pvalue}; z-score: {z}'


@pytest.fixture
def statistics():
    data = {
        'num_dims': NUM_DIMS,
        'loc1': GROUP_LOC1,
        'loc2': GROUP_LOC2,
        'scale1': GROUP_SCALE1,
        'scale2': GROUP_SCALE2,
        'propensity': PROPENSITY,
        'n1': N1,
        'n2': N2,
    }

    model = stan.build("""
    functions {
        %(functions)s
    }

    data {
        int<lower=1> num_dims;
        vector[num_dims] loc1, loc2;
        real<lower=0> scale1, scale2;
        real<lower=0> propensity;
        int<lower=0> n1, n2;
    }

    parameters {
        real dummy;
    }

    transformed parameters {
        // Kernel statistics.
        real<lower=0> mean = evaluate_mean(loc1, loc2, scale1, scale2, propensity);
        real<lower=0> square = evaluate_square(loc1, loc2, scale1, scale2, propensity);
        real cross = evaluate_cross(loc1, loc2, scale1, scale2, propensity);

        // Connection volume statistics.
        real<lower=0> aggregate_mean_intra = evaluate_aggregate_mean(loc1, loc1, scale1, scale1,
                                                                     propensity, n1, 0);
        real<lower=0> aggregate_mean_inter = evaluate_aggregate_mean(loc1, loc2, scale1, scale2,
                                                                     propensity, n1, n2);
        real<lower=0> aggregate_var_intra = evaluate_aggregate_var(loc1, loc1, scale1, scale1,
                                                                   propensity, n1, 0);
        real<lower=0> aggregate_var_inter = evaluate_aggregate_var(loc1, loc2, scale1, scale2,
                                                                   propensity, n1, n2);
    }

    model {
        dummy ~ normal(0, 1);
    }
    """ % {'functions': alsm.model.FUNCTIONS['__all__']}, data=data)
    fit = model.sample(num_chains=1, num_samples=1, num_warmup=0)
    fit.data = data
    return fit


def test_evaluate_mean(statistics: stan.fit.Fit):
    _bootstrap(KERNEL_XY, statistics['mean'])
    mean = alsm.evaluate_mean(
        statistics.data['loc1'], statistics.data['loc2'], statistics.data['scale1'],
        statistics.data['scale2'], statistics.data['propensity'],
    )
    np.testing.assert_allclose(statistics['mean'], mean)


def test_evaluate_square(statistics: stan.fit.Fit):
    _bootstrap(KERNEL_XY ** 2, statistics['square'])
    square = alsm.evaluate_square(
        statistics.data['loc1'], statistics.data['loc2'], statistics.data['scale1'],
        statistics.data['scale2'], statistics.data['propensity'],
    )
    np.testing.assert_allclose(statistics['square'], square)


def test_evaluate_cross(statistics: stan.fit.Fit):
    _bootstrap(KERNEL_XY * KERNEL_XYp, statistics['cross'])
    cross = alsm.evaluate_cross(
        statistics.data['loc1'], statistics.data['loc2'], statistics.data['scale1'],
        statistics.data['scale2'], statistics.data['propensity'],
    )
    np.testing.assert_allclose(statistics['cross'], cross)


def test_evaluate_aggregate_mean_intra(statistics: stan.fit.Fit):
    _bootstrap(ARD_INTRA, statistics['aggregate_mean_intra'])
    aggregate_mean_intra = alsm.evaluate_aggregate_mean(
        statistics.data['loc1'], statistics.data['loc1'], statistics.data['scale1'],
        statistics.data['scale1'], statistics.data['propensity'], statistics.data['n1'], None,
    )
    np.testing.assert_allclose(statistics['aggregate_mean_intra'], aggregate_mean_intra)


def test_evaluate_aggregate_mean_inter(statistics: stan.fit.Fit):
    _bootstrap(ARD_INTER, statistics['aggregate_mean_inter'])
    aggregate_mean_inter = alsm.evaluate_aggregate_mean(
        statistics.data['loc1'], statistics.data['loc2'], statistics.data['scale1'],
        statistics.data['scale2'], statistics.data['propensity'], statistics.data['n1'],
        statistics.data['n2'],
    )
    np.testing.assert_allclose(statistics['aggregate_mean_inter'], aggregate_mean_inter)


def test_evaluate_aggregate_var_intra(statistics: stan.fit.Fit):
    _bootstrap(ARD_INTRA, statistics['aggregate_var_intra'], func=np.var)
    aggregate_var_intra = alsm.evaluate_aggregate_var(
        statistics.data['loc1'], statistics.data['loc1'], statistics.data['scale1'],
        statistics.data['scale1'], statistics.data['propensity'], statistics.data['n1'], None,
    )
    np.testing.assert_allclose(statistics['aggregate_var_intra'], aggregate_var_intra)


def test_evaluate_aggregate_var_inter(statistics: stan.fit.Fit):
    _bootstrap(ARD_INTER, statistics['aggregate_var_inter'], func=np.var)
    aggregate_var_inter = alsm.evaluate_aggregate_var(
        statistics.data['loc1'], statistics.data['loc2'], statistics.data['scale1'],
        statistics.data['scale2'], statistics.data['propensity'], statistics.data['n1'],
        statistics.data['n2'],
    )
    np.testing.assert_allclose(statistics['aggregate_var_inter'], aggregate_var_inter)


def test_group_model_from_data():
    data = alsm.generate_data(np.asarray([10, 20, 30, 40]), 3)
    data['epsilon'] = 1e-6
    posterior = stan.build(alsm.model.GROUP_MODEL, data=data)
    fit = posterior.sample(num_chains=4, num_samples=3, num_warmup=17)
    assert fit.num_chains == 4


def test_group_model_from_group_data():
    data = alsm.generate_group_data(np.asarray([10, 20, 30, 40]), 3)
    data['epsilon'] = 1e-6
    posterior = stan.build(alsm.model.GROUP_MODEL, data=data)
    fit = posterior.sample(num_chains=4, num_samples=3, num_warmup=17)
    assert fit.num_chains == 4


@pytest.fixture
def dummy_fit():
    posterior = stan.build(
        'data { int n; } '
        'parameters { vector[n] x; real y; } '
        'model { x ~ normal(0, 1); y ~ normal(0, 1); }',
        data={'n': 10},
    )
    return posterior.sample(num_chains=3, num_samples=17)


def test_get_samples(dummy_fit):
    for flatten_chains, squeeze in it.product([True, False], [True, False]):
        trailing_shape = (17 * 3,) if flatten_chains else (17, 3)
        x = alsm.get_samples(dummy_fit, 'x', flatten_chains, squeeze)
        assert x.shape == (10,) + trailing_shape
        y = alsm.get_samples(dummy_fit, 'y', flatten_chains, squeeze)
        assert y.shape == (trailing_shape if squeeze else (1,) + trailing_shape)


def test_get_chain(dummy_fit):
    chain = alsm.get_chain(dummy_fit, 'best')
    median_lp = np.median(alsm.get_samples(dummy_fit, 'lp__', False), axis=0)
    assert median_lp.shape == (3,)
    assert np.all(median_lp <= np.median(chain['lp__']))
    assert chain['x'].shape == (10, 17)
    assert chain['y'].shape == (17,)


def test_group_scale_change_of_variables(figure):
    data = {'num_dims': 2, 'alpha': 5, 'beta': 2}
    posterior = stan.build("""
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
    """ % alsm.model.FUNCTIONS, data=data)
    fit = posterior.sample(num_samples=1000)

    # Get the samples and thin them because the samples may still have autocorrelation (which will
    # mess with the bootstrap estimate).
    xs = alsm.get_samples(fit, 'group_scale', flatten_chains=False)
    assert xs.shape == (1000, 4)
    xs = xs[::10].ravel()

    gs = figure.add_gridspec(2, 2)

    ax = figure.add_subplot(gs[:, 0])
    ax.hist(xs, bins=20, density=True)
    lin = np.linspace(0, xs.max(), 100)
    ax.plot(lin, stats.gamma(data['alpha'], scale=1 / data['beta']).pdf(lin), color='C1')
    ax.set_xlabel(r'$\eta\sim\mathrm{Gamma}(%(alpha).1f, %(beta).1f)$' % data)

    ax = figure.add_subplot(gs[0, 1])
    ax.set_xlabel('bootstrapped mean')
    _bootstrap(xs, data['alpha'] / data['beta'], ax=ax)

    ax = figure.add_subplot(gs[1, 1])
    ax.set_xlabel('bootstrapped var')
    _bootstrap(xs, data['alpha'] / data['beta'] ** 2, func=np.var, ax=ax)
