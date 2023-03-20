---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import alsm
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import special, stats


mpl.rcParams['figure.dpi'] = 144
SEED = 3
num_dims = 2
weighted = False
```

# Kernel properties and beta-binomial approximation

```{code-cell} ipython3
np.random.seed(SEED)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax = ax1
# Evaluate the kernel over a range of scales and separations.
ax.set_yscale('log')
dx = np.linspace(0, 3, 100)
delta = np.stack([dx, *np.zeros((num_dims - 1, 100))]).T
scale = np.logspace(-2, 1, 101)

mean = alsm.evaluate_mean(np.zeros(num_dims), delta[:, None], scale, scale, 1).T
mappable = ax.pcolormesh(*np.meshgrid(dx, scale), mean, shading='auto', rasterized=True, vmin=0,
                         vmax=1)
cb = fig.colorbar(mappable, ax=ax, location='top')
cb.set_label(r'Expected kernel $\left\langle\lambda\right\rangle_z$')

levels = [0.05, 0.15, 0.3, 0.6, 0.9]
cs = ax.contour(dx, scale, mean, colors='w', levels=levels, linestyles='--')
plt.clabel(cs)

# Plot the maximum last.
ax.autoscale(False)
argmax_delta = (num_dims * (1 + 2 * scale ** 2)) ** .5
ax.plot(argmax_delta, scale, color='C1')

ax.set_ylabel(r'Scale $\sigma$')
ax.set_xlabel(r'Separation $\delta$')
ax.text(0.05, 0.95, '(a)', va='top', transform=ax.transAxes, color='w')

# Show simulated data.
ax = ax2
num_samples = 100000
group_sizes = np.asarray([10, 15])
means = np.asarray([[0, 0], [1, 0]])
scale = 5
x, y = [np.random.normal(mean, scale, (num_samples, size, 2)) for mean, size
        in zip(means, group_sizes)]
rate = alsm.evaluate_kernel(x[:, :, None], y[:, None], 1)
if weighted:
    agg = np.random.poisson(rate.sum(axis=(1, 2)))
else:
    agg = np.random.binomial(1, rate).sum(axis=(1, 2))
counts = np.bincount(agg) / agg.size
ax.bar(np.arange(counts.size), counts, color='silver', width=1, label='simulated $Y_{ab}$')

# Evaluate properties of the negative/beta binomial approximation.
mean = alsm.evaluate_aggregate_mean(*means, scale, scale, 1, *group_sizes)
var = alsm.evaluate_aggregate_var(*means, scale, scale, 1, *group_sizes, weighted)
if weighted:
    dist = stats.nbinom(*alsm.evaluate_negative_binomial_np(mean, var))
    label = 'negative binomial\napproximation'
else:
    trials = np.prod(group_sizes)
    dist = stats.betabinom(trials, *alsm.evaluate_beta_binomial_ab(trials, mean, var))
    label = 'beta binomial\napproximation'
lin = np.arange(agg.min(), agg.max() + 1)
ax.plot(lin, dist.pmf(lin), label=label)

ax.set_xlabel('Aggregate connections $Y_{ab}$')
ax.set_ylabel('$P(Y_{ab})$')
ax.yaxis.major.formatter.useMathText = True
ax.yaxis.major.formatter.set_powerlimits((0, 0))
ax.text(0.05, 0.95, '(b)', va='top', transform=ax.transAxes)
lines = [
    r'$\delta=1$',
    fr'$\sigma={scale}$',
    f'$n_a={group_sizes[0]}$',
    f'$n_b={group_sizes[1]}$',
]
ax.text(0.95, 0.5, '\n'.join(lines), va='center', ha='right', transform=ax.transAxes)
ax.legend(loc='upper right', fontsize='small')

fig.tight_layout()
fig.savefig('../workspace/kernel.pdf')
```

# Change of variables for group scales

```{code-cell} ipython3
# Show the transformation between eta and scales.
eta = special.expit(5 * np.linspace(-1, 1, 100))
scale = np.sqrt((eta ** (-2 / num_dims) - 1) / 2)
jac = eta ** (- 2 / num_dims - 1) / (2 * num_dims * scale)

fig, ax = plt.subplots()
ax.plot(eta, scale, label=r'Group scale $\sigma$')
ax.plot(eta, jac, label=r'Jacobian $\left\vert\frac{d\sigma}{d\eta}\right\vert$')
ax.set_yscale('log')
ax.set_xlabel(r'Fraction of realised density $\eta$')
ax.legend(fontsize='small')
fig.tight_layout()
```

# Kernel evaluation

```{code-cell} ipython3
np.random.seed(7)
group_locs = np.random.normal(0, 1, 3)
group_scales = np.random.gamma(10, .1, 3)

x, y, z = np.transpose(group_locs + group_scales * np.random.normal(0, 1, (10000, 3)))

d1 = x - y
d2 = x - z
sq = d1 ** 2 + d2 ** 2

cov = np.cov(d1, d2)

sa, sb, sc = group_scales
theory_cov = np.asarray([
    [sa ** 2 + sb ** 2, sa ** 2],
    [sa ** 2, sa ** 2 + sc ** 2],
])

val = (sb ** 2 - sc ** 2 - np.sqrt(4 * sa ** 4 + (sb ** 2 - sc ** 2) ** 2)) / (2 * sa ** 2)
evecs = np.transpose([
    [1, -val], [val, 1],
])
evecs /= np.linalg.norm(evecs, axis=0, keepdims=True)

print('empirical covariance')
print(cov)
print('theoretical covariance')
print(theory_cov)
print('evecs')
print(evecs)


# _, evecs = np.linalg.eig(theory_cov)
zeta, xi = ((np.asarray([d1, d2]).T @ evecs)).T

kwargs = {'alpha': .5, 'marker': '.'}
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.scatter(d1, d2, **kwargs)
ax1.set_xlabel(r'$\delta_{ij}=z_i-z_j$')
ax1.set_ylabel(r'$\delta_{il}=z_i-z_l$')
ax1.set_aspect('equal')

ax2.scatter(zeta, xi, **kwargs)
ax2.set_xlabel(r'$\zeta$')
ax2.set_ylabel(r'$\xi$')
ax2.set_aspect('equal')
fig.tight_layout()

np.testing.assert_allclose(sq, (zeta ** 2 + xi ** 2))
```

# Correlation of aggregate connection volumes

```{code-cell} ipython3
# Consider correlations in volumes.
num_samples = 5000
group_sizes = np.asarray([30, 100, 110])
group_locs = np.asarray([[0, 0], [.1, 0], [-1, 0]])
scale = .5
x, y, z = [np.random.normal(mean, scale, (num_samples, size, 2)) for mean, size
           in zip(group_locs, group_sizes)]

rate0 = alsm.evaluate_kernel(x[:, :, None], x[:, None], 1)
i = np.arange(group_sizes[0])
rate0[..., i, i] = 0
rate1 = alsm.evaluate_kernel(x[:, :, None], y[:, None], 1)
rate2 = alsm.evaluate_kernel(x[:, :, None], z[:, None], 1)
rates = [rate0, rate1, rate2]

if weighted:
    aggs = [np.random.poisson(rate.sum(axis=(1, 2))) for rate in rates]
else:
    aggs = [np.random.binomial(1, rate).sum(axis=(1, 2)) for rate in rates]
agg0, agg1, agg2 = aggs

# Evaluate the variances and covariance.
means = [
    alsm.evaluate_aggregate_mean(
        group_locs[0], group_locs[0], scale, scale, 1, group_sizes[0], None),
    alsm.evaluate_aggregate_mean(
        group_locs[0], group_locs[1], scale, scale, 1, group_sizes[0], group_sizes[1]),
    alsm.evaluate_aggregate_mean(
        group_locs[0], group_locs[2], scale, scale, 1, group_sizes[0], group_sizes[2])
]
variances = [
    alsm.evaluate_aggregate_var(
        group_locs[0], group_locs[0], scale, scale, 1, group_sizes[0], None, weighted),
    alsm.evaluate_aggregate_var(
        group_locs[0], group_locs[1], scale, scale, 1, group_sizes[0], group_sizes[1], weighted),
    alsm.evaluate_aggregate_var(
        group_locs[0], group_locs[2], scale, scale, 1, group_sizes[0], group_sizes[2], weighted),
]
covs = [
    alsm.evaluate_aggregate_cov(
        *group_locs[:2], None, scale, scale, None, 1, *group_sizes[:2], None),
    alsm.evaluate_aggregate_cov(*group_locs, scale, scale, scale, 1, *group_sizes)
]
# expected_cov = np.asarray([[var1, cov], [cov, var2]])

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

for (ax, a, b, ma, mb, va, vb, cov) in [
    (ax1, agg1, agg0, means[1], means[0], variances[1], variances[0], covs[0]),
    (ax2, agg1, agg2, means[1], means[2], variances[1], variances[2], covs[1]),
]:
    ax.scatter(a, b, marker='.', alpha=.25)
    expected_cov = np.asarray([[va, cov], [cov, vb]])

    print('observed')
    print(np.round(np.cov(a, b), 3))
    print('predicted')
    print(np.round(expected_cov, 3))

    # Show the covariance matrix.
    evals, evecs = np.linalg.eigh(expected_cov)
    radii = 2 * np.sqrt(evals)
    if np.isfinite(radii).all():
        angle = - np.rad2deg(np.arctan2(*evecs[:, 1]))
        ellipse = mpl.patches.Ellipse((ma, mb), *(2 * radii), angle=angle,
                                      facecolor='none', edgecolor='C1')
        ax.add_patch(ellipse)
    ax.scatter(ma, mb, marker='X', color='C1').set_edgecolor('w')
    ax.set_xlabel('$Y_{ab}$')

ax1.set_ylabel('$Y_{aa}$')
ax2.set_ylabel('$Y_{ac}$')

fig.tight_layout()
```
