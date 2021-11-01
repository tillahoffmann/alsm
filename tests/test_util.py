from alsm import util as alsm_util
from alsm import model as alsm_model
import itertools as it
import numpy as np
import pytest
from scipy import stats
import stan


def test_evaluate_grouping_matrix():
    num_groups = 10
    num_nodes = 100
    idx = np.random.randint(num_groups, size=num_nodes)
    grouping = alsm_util.evaluate_grouping_matrix(idx, num_groups)
    # Ensure each node belongs to exactly one group.
    np.testing.assert_array_equal(grouping.sum(axis=0), 1)
    # Ensure that the groups add up.
    counts = np.bincount(idx, minlength=num_groups)
    np.testing.assert_array_equal(grouping.sum(axis=1), counts)


def test_plot_edges():
    data = alsm_model.generate_data(np.asarray([10, 20]), 2, population_scale=.1, propensity=1)
    collection = alsm_util.plot_edges(data['locs'], data['adjacency'])
    assert collection.get_alpha().size == (data['adjacency'] > 0).sum()


def test_align_samples():
    reference = np.random.normal(0, 1, (10, 2))
    offsets = np.random.normal(0, 1, (2, 20))
    offsets[:, 0] = 0
    # Construct rotation matrices with shape (2, 2, num_samples).
    angles = np.random.uniform(0, 2 * np.pi, 20)
    angles[0] = 0
    rotations = np.asarray([
        [np.cos(angles), -np.sin(angles)],
        [np.sin(angles), np.cos(angles)]
    ])
    # Get modified samples.
    samples = (reference @ rotations + offsets[:, None, :]).T
    # Align the samples and compare with reference.
    aligned = alsm_util.align_samples(samples)
    for x in aligned:
        np.testing.assert_allclose(x, reference - reference.mean(axis=0))


def test_negbinom_mean_var_to_params():
    mean = np.random.gamma(1, 1)
    var = mean + np.random.gamma(1, 1)

    dist = stats.nbinom(*alsm_util.negative_binomial_np(mean, var))
    np.testing.assert_allclose(mean, dist.mean())
    np.testing.assert_allclose(var, dist.var())


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
        x = alsm_util.get_samples(dummy_fit, 'x', flatten_chains, squeeze)
        assert x.shape == (10,) + trailing_shape
        y = alsm_util.get_samples(dummy_fit, 'y', flatten_chains, squeeze)
        assert y.shape == (trailing_shape if squeeze else (1,) + trailing_shape)


def test_get_chain(dummy_fit):
    chain = alsm_util.get_chain(dummy_fit, 'best')
    median_lp = np.median(alsm_util.get_samples(dummy_fit, 'lp__', False), axis=0)
    assert median_lp.shape == (3,)
    assert np.all(median_lp <= np.median(chain['lp__']))
    assert chain['x'].shape == (10, 17)
    assert chain['y'].shape == (17,)
