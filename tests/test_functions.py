import alsm
import alsm.stan
import numpy as np
import pytest
import stan


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
    """ % {'functions': alsm.stan.FUNCTIONS['__all__']}, data=data)
    return model.sample(num_chains=1, num_samples=1, num_warmup=0)


def _bootstrap(xs, x, func=np.mean):
    """
    Bootstrap the mean of `xs` and compare with the theoretical value `x`.
    """
    assert xs.shape == (NUM_SAMPLES,)
    # Evaluate the bootstrap samples.
    bootstrap = np.asarray([
        func(xs[np.random.randint(NUM_SAMPLES, size=NUM_SAMPLES)], axis=0)
        for _ in range(NUM_BOOTSTRAP)
    ])

    # Evaluate the pvalue and z-score.
    pvalue = (x < bootstrap).mean()
    pvalue = min(pvalue, 1 - pvalue)
    assert pvalue > 1 / NUM_BOOTSTRAP
    z = (x - xs.mean()) / bootstrap.std()  # noqa


def test_evaluate_mean(statistics):
    _bootstrap(KERNEL_XY, statistics['mean'])


def test_evaluate_square(statistics):
    _bootstrap(KERNEL_XY ** 2, statistics['square'])


def test_evaluate_cross(statistics):
    _bootstrap(KERNEL_XY * KERNEL_XYp, statistics['cross'])


def test_evaluate_aggregate_mean_intra(statistics):
    _bootstrap(ARD_INTRA, statistics['aggregate_mean_intra'])


def test_evaluate_aggregate_mean_inter(statistics):
    _bootstrap(ARD_INTER, statistics['aggregate_mean_inter'])


def test_evaluate_aggregate_var_intra(statistics):
    _bootstrap(ARD_INTRA, statistics['aggregate_var_intra'], func=np.var)


def test_evaluate_aggregate_var_inter(statistics):
    _bootstrap(ARD_INTER, statistics['aggregate_var_inter'], func=np.var)
