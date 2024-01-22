from alsm import util as alsm_util
from alsm import model as alsm_model
import cmdstanpy
import numpy as np
import os
import pytest
import tempfile


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
    data = alsm_model.generate_data(np.asarray([10, 20]), 2, population_scale=.1, propensity=1,
                                    weighted=False)
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


@pytest.fixture
def dummy_fit() -> cmdstanpy.CmdStanMCMC:
    posterior = cmdstanpy.CmdStanModel(stan_file=alsm_util.write_stanfile(
        'data { int n; } '
        'parameters { vector[n] x; real y; } '
        'model { x ~ normal(0, 1); y ~ normal(0, 1); }',
    ))
    return posterior.sample(chains=3, iter_sampling=17, data={'n': 10})


def test_get_samples(dummy_fit: cmdstanpy.CmdStanMCMC):
    for flatten_chains in [True, False]:
        trailing_shape = (17 * 3,) if flatten_chains else (17, 3)
        x = alsm_util.get_samples(dummy_fit, 'x', flatten_chains)
        assert x.shape == (10,) + trailing_shape
        y = alsm_util.get_samples(dummy_fit, 'y', flatten_chains)
        assert y.shape == trailing_shape


def test_get_chain(dummy_fit: cmdstanpy.CmdStanMCMC):
    chain = alsm_util.get_chain(dummy_fit, 'best')
    median_lp = np.median(alsm_util.get_samples(dummy_fit, 'lp__', False), axis=0)
    assert median_lp.shape == (3,)
    assert np.all(median_lp <= np.median(chain['lp__']))
    assert chain['x'].shape == (10, 17)
    assert chain['y'].shape == (17,)


def test_rotation():
    x = [1, 0]
    y = alsm_util.evaluate_rotation_matrix(np.pi / 2) @ x
    np.testing.assert_allclose(y, [0, 1], atol=1e-9)

    y = alsm_util.evaluate_rotation_matrix(3 * np.pi / 4) @ x
    np.testing.assert_allclose(y, np.asarray([-1, 1]) / np.sqrt(2))


def test_estimate_mode(figure):
    # Generate an arc of points and ensure that the centre is roughly in the right place.
    n = 500

    x = np.concatenate([
        np.random.normal([1, 1], .1, size=(n, 2)),
        np.random.normal([-1, -1], .01, size=(n, 2)),
    ])

    ax = figure.add_subplot()
    ax.scatter(*x.T, label='samples')
    ax.set_aspect('equal')
    ax.scatter(*x.mean(axis=0), label='naive')
    mode = alsm_util.estimate_mode(x)
    ax.scatter(*mode, label='estimate')
    ax.legend(loc='best', fontsize='small')

    np.testing.assert_array_less(mode, -.5)


@pytest.mark.parametrize('batch_shape', [(10,), (3, 5)])
def test_estimate_mode_batch(batch_shape):
    x = np.random.normal(0, 1, size=(batch_shape) + (50, 3))
    mode = alsm_util.estimate_mode(x)
    assert mode.shape == batch_shape + (3,)


def test_invert_index():
    x = np.random.normal(0, 1, 27)
    index = np.random.permutation(x.size)
    y = x[index]
    inverted = alsm_util.invert_index(index)
    np.testing.assert_array_equal(x, y[inverted])


def test_write_stanfile():
    with tempfile.TemporaryDirectory() as directory:
        filename1 = alsm_util.write_stanfile("model code", directory=directory)
        mtime1 = os.path.getmtime(filename1)
        filename2 = alsm_util.write_stanfile("model code", directory=directory)
        mtime2 = os.path.getmtime(filename2)

        assert filename1 == filename2
        assert mtime1 == mtime2


def test_get_elbo():
    code = """
        data {
            int<lower=0> n;
            array [n] int<lower=0, upper=1> y;
        }

        parameters {
            real<lower=0,upper=1> theta;
        }

        model {
            theta ~ beta(1,1);
            y ~ bernoulli(theta);
        }
    """
    model = cmdstanpy.CmdStanModel(stan_file=alsm_util.write_stanfile(code))
    approx = model.variational(data={'n': 100, 'y': np.random.binomial(1, 0.7, 100)})
    elbo = alsm_util.get_elbo(approx)
    assert np.isfinite(elbo)
