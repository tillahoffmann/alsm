---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import alsm
import cmdstanpy
import itertools as it
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy import stats
import os
import pickle


mpl.rcParams["figure.dpi"] = 144
SMOKE_TEST = "CI" in os.environ
SEED = int(os.environ.get("SEED", "7"))
NUM_GROUPS = int(os.environ.get("NUM_GROUPS", "10"))
NUM_DIMS = int(os.environ.get("NUM_DIMS", "2"))
SCALE_PRIOR_SCALE = float(os.environ.get("SCALE_PRIOR_SCALE", "1"))
SCALE_PRIOR_TYPE = os.environ.get("SCALE_PRIOR_TYPE", "cauchy")
OUTPUT = os.environ.get("OUTPUT", f"simulation-{SCALE_PRIOR_TYPE}-{SCALE_PRIOR_SCALE}.pkl")
```

```{code-cell} ipython3
np.random.seed(SEED)
group_sizes = np.random.poisson(100, NUM_GROUPS)
data = alsm.generate_data(
    group_sizes,
    NUM_DIMS,
    weighted=False,
    group_scales=np.random.gamma(3, 1 / 5, NUM_GROUPS),
    population_scale=2.5,
    propensity=0.1,
)

# Plot the detailed network.
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.scatter(*data['locs'].T, c=data['group_idx'], cmap='tab10', marker='.')
alsm.plot_edges(data['locs'], data['adjacency'], ax=ax1, alpha=.2, zorder=0)
ax1.set_aspect('equal')

# Plot the aggregate network and the radius of clusters.
pts = ax2.scatter(*data['group_locs'].T, c=np.arange(NUM_GROUPS), cmap='tab10')
alsm.plot_edges(data['group_locs'], data['group_adjacency'], ax=ax2, zorder=0, alpha_min=.1)
ax2.set_aspect('equal')

plt.draw()
for color, group_loc, group_scale in zip(pts.get_facecolors(), data['group_locs'],
                                         data['group_scales']):
    circle = mpl.patches.Circle(group_loc, 2 * group_scale, color=color, alpha=.25)
    ax2.add_patch(circle)

ax2.autoscale_view()

print(f'mean degree: {data["group_adjacency"].sum() / data["num_nodes"]:.3f}')
print('ratio of largest to smallest scale: '
      f'{data["group_scales"].max() / data["group_scales"].min():.3f}')

# Show the colormap so we know which group is which.
mpl.cm.tab10
```

```{code-cell} ipython3
pairs = list(it.combinations(range(NUM_GROUPS), 2))

var = np.nan * np.ones(len(pairs))
cov = np.nan * np.ones_like(var)[:, None] * np.ones(NUM_GROUPS)
corr = np.nan * np.ones_like(cov)
group_locs = data['group_locs']
group_scales = data['group_scales']
propensity = data['propensity']
weighted = data['weighted']

for i, (a, b) in enumerate(pairs):
    var[i] = var_ab = alsm.evaluate_aggregate_var(
        group_locs[a], group_locs[b], group_scales[a], group_scales[b],
        propensity, group_sizes[a], group_sizes[b], weighted
    )
    for c in range(NUM_GROUPS):
        var_ac = alsm.evaluate_aggregate_var(
            group_locs[a], group_locs[c], group_scales[a], group_scales[c],
            propensity, group_sizes[a], group_sizes[c], weighted
        )

        # If b and c are the same, we are just looking at the variance Y_{ab}Y_{ab}
        if b == c:
            cov_ab_ac = var_ab
            cov_ab_ac = np.nan
        # If a and c are the same, we're looking at the intra-inter covariance Y_{ab}Y_{aa}
        # which we haven't figured out yet.
        elif a == c:
            cov_ab_ac = alsm.evaluate_aggregate_cov(
                group_locs[a], group_locs[b], None,
                group_scales[a], group_scales[b], None,
                propensity, group_sizes[a], group_sizes[b], None,
            )
        # If b and c are different, we're looking at the covariance.
        else:
            cov_ab_ac = alsm.evaluate_aggregate_cov(
                group_locs[a], group_locs[b], group_locs[c],
                group_scales[a], group_scales[b], group_scales[c],
                propensity, group_sizes[a], group_sizes[b], group_sizes[c],
            )
        cov[i, c] = cov_ab_ac
        corr[i, c] = cov_ab_ac / np.sqrt(var_ab * var_ac)


# Set the size of the group color index indicator.
size = 2
fig, ax = plt.subplots()
ax.patch.set_facecolor('k')
vmax = np.nanmax(np.abs(corr))
cmap = mpl.colormaps.get_cmap('coolwarm').copy()
cmap.set_bad('w')
im = ax.imshow(corr.T, vmax=vmax, vmin=-vmax, cmap=cmap)
ax.imshow(np.transpose(pairs), cmap='tab10', extent=(- .5, len(pairs) - .5, - .5, -2 * size - .5))
ax.imshow(np.arange(NUM_GROUPS)[:, None], cmap='tab10',
          extent=(-size - .5, - .5, NUM_GROUPS - .5, - .5))
ax.set_ylim(NUM_GROUPS - .5, -2 * size - .5)
ax.set_xlim(-size - .5, len(pairs) - .5)
ax.xaxis.tick_top()
ax.set_xlabel('Combined group index $ab$')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Group index $c$')
ax.set_xticks([])
ax.set_yticks([])

for fn in [ax.axhline, ax.axvline]:
    fn(-0.5, color='w', ls=':')

cb = fig.colorbar(im, ax=ax, location='bottom', pad=0.05)
cb.set_label(r'Correlation $\mathrm{corr}\left(Y_{ab},Y_{ac}\right)$')
fig.tight_layout()
fig.savefig('../workspace/correlation.pdf')
fig.savefig('../workspace/correlation.png')
```

```{code-cell} ipython3
# Apply a permutation so "informative" groups pin down the posterior.
index = np.arange(NUM_GROUPS)
pinned = [6, 9]
for i, j in enumerate(pinned):
    index[i] = j
    index[j] = i
# Make sure the index is fine.
np.testing.assert_array_equal(np.sort(index), np.arange(NUM_GROUPS))

data['epsilon'] = 1e-20

# Fit the model.
model_code = alsm.get_group_model_code(
    scale_prior_type=SCALE_PRIOR_TYPE,
    scale_prior_scale=SCALE_PRIOR_SCALE,
)
posterior = cmdstanpy.CmdStanModel(stan_file=alsm.write_stanfile(model_code))
fit = posterior.sample(
    alsm.apply_permutation_index(data, index),
    iter_warmup=10 if SMOKE_TEST else None,
    iter_sampling=27 if SMOKE_TEST else None,
    chains=3 if SMOKE_TEST else 24,
    seed=SEED,
    inits=1e-2,
    show_progress=False,
)
```

```{code-cell} ipython3
print(model_code)
```

```{code-cell} ipython3
# Go through each chain and evaluate the log probability as well as the alignment score with the
# original data.

median_losses = []
metrics = []
inverse = alsm.invert_index(index)
method_variables = fit.method_variables()
divergent = method_variables["divergent__"].mean(axis=0)
stepsize = method_variables["stepsize__"].mean(axis=0)
for i in range(fit.chains):
    chain = alsm.apply_permutation_index(alsm.get_chain(fit, i), inverse)
    median_lp = np.median(chain['lp__'])

    # Align the samples.
    samples = np.rollaxis(chain['group_locs'], -1)
    aligned = alsm.align_samples(samples)

    # Align the samples to the reference data.
    reference = data['group_locs'] - data['group_locs'].mean(axis=0)
    transform, _ = orthogonal_procrustes(np.mean(aligned, axis=0), reference)
    aligned = aligned @ transform

    # Compute the median loss.
    median_loss = np.median((aligned - reference) ** 2)

    # Information criterion.
    elppd = alsm.util.evaluate_elppd(chain["log_likelihood"])

    print("; ".join([
        f'chain {i}',
        f'median lp: {median_lp:.3f}',
        f'median loss: {median_loss:.3f}',
        f'divergent: {divergent[i]:.3f}',
        f'stepsize: {stepsize[i]:.3g}',
        f'elppd: {elppd:.3f}',
    ]))

    median_losses.append(median_loss)
    # Ignore chains where more than half the samples have diverged.
    metrics.append(-1e9 if divergent[i] > 0.5 or stepsize[i] < 1e-4 else median_lp)

median_losses = np.asarray(median_losses)
metrics = np.asarray(metrics)
best_chain = np.argmax(metrics)
print(f"best chain: {metrics[best_chain]}")
chain = alsm.apply_permutation_index(alsm.get_chain(fit, best_chain), inverse)
best_chain
```

```{code-cell} ipython3
angle = 80
rotation = alsm.evaluate_rotation_matrix(np.deg2rad(angle))

locs = data['locs'] @ rotation
samples = np.rollaxis(chain['group_locs'], -1) @ rotation
reference = data['group_locs'] @ rotation

locs = locs[..., ::-1]
samples = samples[..., ::-1]
reference = reference[..., ::-1]

fig = plt.figure(figsize=(6, 5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect('equal')
alsm.plot_edges(locs, data['adjacency'], zorder=0, ax=ax1, alpha=.25)
# We'll order the points by the scale of the group they belong to such that groups
# with small spatial scatter are "on top".
idx = np.argsort(-data['group_scales'][data['group_idx']])
ax1.scatter(*locs[idx].T, c=data['group_idx'][idx], cmap='tab10', marker='.')
ax1.set_xlabel('Embedding $z_1$')
ax1.set_ylabel('Embedding $z_2$')

# Align the samples with one another and then with the reference.
samples = alsm.align_samples(samples)
modes = alsm.estimate_mode(np.rollaxis(samples, 1))
transform, _ = orthogonal_procrustes(modes, reference - reference.mean(axis=0))
samples = (samples @ transform) + reference.mean(axis=0)
modes = (modes @ transform) + reference.mean(axis=0)

ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
alsm.plot_edges(modes, data['group_adjacency'] ** .5, zorder=0, ax=ax2, lw=3)
c = np.arange(data['num_groups'])[:, None] * np.ones(samples.shape[0])
ax2.scatter(*samples.T, c=c, cmap='tab10', marker='.', alpha=.05)

# Show the scales.
factor = 2
for i, (xy, radius) in enumerate(zip(modes, np.median(chain['group_scales'], axis=-1))):
    # Slightly desaturate the colour so the circle is visible against the background of samples.
    color = [c * 0.7 for c in mpl.colors.to_rgb(f'C{i}')]
    circle = mpl.patches.Circle(xy, factor * radius, edgecolor=color, facecolor='none')
    ax2.add_patch(circle)

ax2.set_aspect('equal')
ax2.set_xlabel(r'Embedding $z_1$')
plt.setp(ax2.yaxis.get_ticklabels(), visible=False)

# Show the propensity plot.
ax3 = fig.add_subplot(gs[1, 0])
x = chain['propensity']
ax3.hist(x, density=True, bins=20, color='silver')
kde = stats.gaussian_kde(x)
lin = np.linspace(x.min(), x.max(), 100)
ax3.plot(lin, kde(lin))
ax3.axvline(data['propensity'], color='k', ls=':')
ax3.set_xlabel(r'Propensity $\theta$')
ax3.set_ylabel(r'Posterior $p(\theta\mid Y)$')
ax3.set_xlim(0.085, 0.13)

# Show the scales.
ax4 = fig.add_subplot(gs[1, 1])
l, m, u = np.percentile(chain['group_scales'], [2.5, 50, 97.5], axis=-1)
x = data['group_scales']
lims = x.min(), x.max()
ax4.plot(lims, lims, color='k', ls=':')
ax4.errorbar(x, m, (m - l, u - m), color='gray', ls='none')
ax4.scatter(x, m, c=np.arange(data['num_groups']), cmap='tab10', zorder=2)
ax4.set_aspect('equal')
ax4.set_xlabel(r'Group scales $\sigma$')
ax4.set_ylabel(r'Inferred group scales')

labels = [
    (ax1, 'top left', '(a)'),
    (ax2, 'top left', '(b)'),
    (ax3, 'top right', '(c)'),
    (ax4, 'top left', '(d)'),
]
for ax, loc, label in labels:
    va, ha = loc.split()
    ax.text(0.05 if ha == 'left' else 0.95, 0.05 if va == 'bottom' else 0.95, label,
            transform=ax.transAxes, ha=ha, va=va)

fig.tight_layout()
fig.savefig(f"../workspace/simulation-{SCALE_PRIOR_TYPE}-{SCALE_PRIOR_SCALE}.pdf")
fig.savefig(f"../workspace/simulation-{SCALE_PRIOR_TYPE}-{SCALE_PRIOR_SCALE}.png")
```

```{code-cell} ipython3
# Show the scatter plot.
fig, ax = plt.subplots()
ax.scatter(*samples.T, cmap='tab10', marker='.', alpha=.1, label='posterior samples',
           c=np.arange(NUM_GROUPS)[:, None] * np.ones(aligned.shape[0]))
pts = ax.scatter(*reference.T, c=np.arange(NUM_GROUPS), marker='X', cmap='tab10', label='reference')
pts.set_edgecolor('w')
pts = ax.scatter(*modes.T, c=np.arange(NUM_GROUPS), marker='o', cmap='tab10', label='modes')
pts.set_edgecolor('w')
ax.set_aspect('equal')
ax.legend(fontsize='small')

fig.tight_layout()
```

```{code-cell} ipython3
# Show the posterior predictive replication.
fig, ax = plt.subplots()

x = data['group_adjacency'].ravel()
ys = chain['ppd_group_adjacency'].reshape(x.shape + (-1,))
l, m, u = np.percentile(ys, [2.5, 50, 97.5], axis=-1)
lims = x.min(), x.max()
ax.plot(lims, lims, color='k', ls=':')
ax.errorbar(x, m, (m - l, u - m), ls='none', marker='.')
ax.set_aspect('equal')
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set_xlabel('Group adjacency $Y$')
ax.set_ylabel('Group adjacency posterior replicates')
fig.tight_layout()
```

```{code-cell} ipython3
if OUTPUT:
    with open(OUTPUT, "wb") as fp:
        pickle.dump({
            "SMOKE_TEST": SMOKE_TEST,
            "SEED": SEED,
            "NUM_GROUPS": NUM_GROUPS,
            "NUM_DIMS": NUM_DIMS,
            "OUTPUT": OUTPUT,
            "best_chain_idx": best_chain,
            "best_chain": chain,
            "data": data,
            "median_losses": np.asarray(median_losses),
            "metrics": np.asarray(metrics),
            "samples": samples,
            "modes": modes,
            "reference": reference,
            "fit": fit,
            "permutation_index": index,
        }, fp)
```
