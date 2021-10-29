import alsm
import numpy as np
from scipy import stats


def test_generate_data():
    group_sizes = np.asarray([10, 20, 30])
    num_groups, = group_sizes.shape
    num_nodes = group_sizes.sum()
    num_dims = 2
    data = alsm.generate_data(group_sizes, num_dims)
    assert data['population_scale'].shape == ()
    assert data['adjacency'].shape == (num_nodes, num_nodes)
    assert data['group_adjacency'].shape == (num_groups, num_groups)
    assert data['adjacency'].sum() == data['group_adjacency'].sum()
    assert data['group_locs'].shape == (num_groups, num_dims)
    assert data['group_scales'].shape == (num_groups,)
    assert data['locs'].shape == (num_nodes, num_dims)


def test_evaluate_grouping_matrix():
    num_groups = 10
    num_nodes = 100
    idx = np.random.randint(num_groups, size=num_nodes)
    grouping = alsm.evaluate_grouping_matrix(idx, num_groups)
    # Ensure each node belongs to exactly one group.
    np.testing.assert_array_equal(grouping.sum(axis=0), 1)
    # Ensure that the groups add up.
    counts = np.bincount(idx, minlength=num_groups)
    np.testing.assert_array_equal(grouping.sum(axis=1), counts)


def test_plot_edges():
    data = alsm.generate_data(np.asarray([10, 20]), 2, population_scale=.1, propensity=1)
    collection = alsm.plot_edges(data['locs'], data['adjacency'])
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
    aligned = alsm.align_samples(samples)
    for x in aligned:
        np.testing.assert_allclose(x, reference - reference.mean(axis=0))


def test_negbinom_mean_var_to_params():
    mean = np.random.gamma(1, 1)
    var = mean + np.random.gamma(1, 1)

    dist = stats.nbinom(*alsm.negative_binomial_np(mean, var))
    np.testing.assert_allclose(mean, dist.mean())
    np.testing.assert_allclose(var, dist.var())
