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
import pickle
import alsm
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes
import gc

np.random.seed(0)
```

# Load the aggregate data

```{code-cell} ipython3
# Load the result and find the best chain.
aggregate_result = pd.read_pickle("../workspace/addhealth-aggregate.pkl")
aggregate_fit = aggregate_result["fit"]
data = aggregate_result["data"]
method_variables = aggregate_fit.method_variables()

best_chain = None
best_metric = - np.inf

for i in range(aggregate_fit.chains):
    chain = alsm.util.get_chain(aggregate_fit, i)
    metric = np.median(chain["lp__"])
    print(i, metric, method_variables["stepsize__"][:, i].mean(axis=0),
          method_variables["divergent__"][:, i].mean(axis=0))
    if metric > best_metric:
        print("##########", i)
        best_metric = metric
        best_chain = chain

# Copied over from addhealth-fit. Should probably pickle it instead.
num_groups = best_chain["group_locs"].shape[0]

best_chain = alsm.apply_permutation_index(best_chain, alsm.invert_index(aggregate_result["permutation_index"]))
aggregate_best_chain = best_chain
del best_chain, chain
```

```{code-cell} ipython3
# We apply a permutation to samples before aligning to ensure our alignment isn't 
# dominated by temporal correlation.
samples = np.moveaxis(aggregate_best_chain["group_locs"], -1, 0)
aligned = alsm.align_samples(samples)
aligned = aligned - aligned.mean(axis=1, keepdims=True)
modes = alsm.estimate_mode(np.moveaxis(aligned, 0, 1))
```

```{code-cell} ipython3
# Align samples and estimate the mode.
fig, ax = plt.subplots()
alsm.plot_edges(modes, data["group_adjacency"], lw=3)

c = aggregate_result["group_attributes"].grade.values[:, None] * np.ones(aggregate_fit.num_draws_sampling)
pts = ax.scatter(*aligned.T, c=c, marker='.', alpha=.01)
plt.draw()

for xy, radius, tup in zip(modes, np.median(aggregate_best_chain['group_scales'], axis=-1),
                           aggregate_result["group_attributes"].itertuples()):
    color = pts.cmap(pts.norm(tup.grade))
    circle = mpl.patches.Circle(xy, radius, facecolor='none', edgecolor=color)
    ax.add_patch(circle)
    ax.scatter(*xy, color=color, marker='s' if tup.sex == 1 else 'o', zorder=2).set_edgecolor('w')

ax.set_aspect('equal')
```

# Load the individual data

```{code-cell} ipython3
filenames = list(Path("../workspace").glob("addhealth-individual-*.pkl"))

best_chain = None
best_metric = -np.inf

for filename in filenames:
    print(filename)
    individual_result = pd.read_pickle(filename)
    individual_fit = individual_result["fit"]
    method_variables = individual_fit.method_variables()

    for i in range(individual_fit.chains):
        chain = alsm.util.get_chain(individual_fit, i)
        metric = np.median(chain["lp__"])
        print(i, metric, method_variables["stepsize__"][:, i].mean(axis=0),
              method_variables["divergent__"][:, i].mean(axis=0))
        if metric > best_metric:
            print("##########", i)
            best_metric = metric
            best_chain = chain
    del individual_result
    gc.collect()

individual_best_chain = best_chain
del best_chain, chain
```

# Generate the plots.

```{code-cell} ipython3
class LegendTitle(mpl.legend_handler.HandlerBase):
    """
    Handler to show a subtitle in a legend (cf. https://stackoverflow.com/a/38486135/1150961).
    """
    def __init__(self, **text_props):
        self.text_props = text_props or {}

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mpl.text.Text(x0, y0, orig_handle, **self.text_props)
        handlebox.add_artist(title)
        return title


adjacency = data["adjacency"]
group_adjacency = data["group_adjacency"]
group_attributes = aggregate_result["group_attributes"]
attributes = aggregate_result["attributes"]

fig = plt.figure()
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
plt.setp(ax1.xaxis.get_ticklabels(), visible=False)
ax3 = fig.add_subplot(gs[:, 1])

# The individual-level fit is x, the group-level fit is y.
rotation = alsm.evaluate_rotation_matrix(np.deg2rad(-110))

xs = np.rollaxis(individual_best_chain["locs"], -1, 0) @ rotation
x = np.rollaxis(individual_best_chain["group_locs"], -1, 0) @ rotation

# xs = alsm.align_samples(xs) @ rotation
# x = alsm.align_samples(x) @ rotation

xs_samples, x_samples = alsm.align_samples(xs, x)


xs = alsm.estimate_mode(np.moveaxis(xs_samples, 0, 1))
x = alsm.estimate_mode(np.moveaxis(x_samples, 0, 1))

# Center both.
x = x - x.mean(axis=0)
xs = xs - x.mean(axis=0)

y = modes
ys = aligned

# Scale the group-level fit.
scale_factor = individual_best_chain["population_scale"].mean() / np.mean(aggregate_best_chain['population_scale'])
y = scale_factor * y
ys = scale_factor * ys

# Apply the rigid procrustes transform.
transform, _ = orthogonal_procrustes(y, x)
y = y @ transform
ys = ys @ transform

alsm.plot_edges(xs, adjacency, alpha=.2, ax=ax1, zorder=0)
alsm.plot_edges(y, group_adjacency, ax=ax2, zorder=1, lw=3)

pts_kwargs = {
    'cmap': mpl.colormaps.get_cmap('viridis'),
    'vmin': group_attributes.grade.min(),
    'vmax': group_attributes.grade.max(),
}

# Show the individuals.
for sex, subset in attributes.groupby('sex'):
    marker = 'so'[sex - 1]
    ax1.scatter(*xs[subset.index].T, c=subset.grade, marker=marker, s=7, **pts_kwargs)

# Show the markers for group location posterior samples and modes.
ax2.scatter(*ys.T, c=group_attributes.grade.values[:, None] * np.ones(aggregate_fit.num_draws_sampling),
            marker='.', alpha=.025, zorder=0, **pts_kwargs)
for sex, subset in group_attributes.groupby('sex'):
    marker = 'so'[sex - 1]
    ax1.scatter(*x[subset.index].T, c=subset.grade, marker=marker, **pts_kwargs).set_edgecolor('w')
    pts = ax2.scatter(*y[subset.index].T, c=subset.grade, marker=marker, zorder=2, **pts_kwargs)
    pts.set_edgecolor('w')

plt.draw()

for xy, radius, tup in zip(y, np.median(aggregate_best_chain['group_scales'], axis=-1) * scale_factor,
                           group_attributes.itertuples()):
    color = pts.cmap(pts.norm(tup.grade))
    circle = mpl.patches.Circle(xy, radius, facecolor='none', edgecolor=color)
    ax2.add_patch(circle)

for ax, label in [(ax1, '(a)'), (ax2, '(b)')]:
    ax.set_aspect('equal')
    ax.text(0.95, 0.95, label, transform=ax.transAxes, va='top', ha='right')
    ax.autoscale_view()
    ax.set_ylabel('Embedding $z_2$')

ax2.set_xlabel('Embedding $z_1$')

# Add a legend for the symbols.
ax = ax2
handle_girls = ax.scatter([], [], color='none', marker='o', s=15)
handle_girls.set_edgecolor('k')
handle_boys = ax.scatter([], [], color='none', marker='s', s=15)
handle_boys.set_edgecolor('k')

handles = [
    'Sex',
    handle_girls,
    handle_boys,
    'Grade',
]
labels = [
    None,
    'female',
    'male',
    None,
]


for grade in sorted(attributes.grade.unique()):
    labels.append(str(grade))
    handles.append(ax.scatter([], [], color=pts.cmap(pts.norm(grade)), marker='.'))

legend = ax.legend(handles, labels, fontsize='x-small', loc='center left', handletextpad=.25,
                   frameon=False, handler_map={str: LegendTitle(fontsize='x-small')})
ax.set_xlim(left=-9.5)

ax = ax3
ax.scatter(aggregate_best_chain['population_scale'], aggregate_best_chain['propensity'], marker='.', label='group model',
           alpha=.5)
ax.scatter(individual_best_chain["population_scale"][::10], individual_best_chain["propensity"][::10], marker='.',
           label='individual\nmodel', alpha=.5)
# Plot the degeneracy line.
num_nodes = attributes.shape[0]
num_dims = y.shape[-1]
density = group_adjacency.sum() / (num_nodes * (num_nodes - 1))
max_scale = np.sqrt((density ** (- 2 / num_dims) - 1) / 2)
lin = np.linspace(individual_best_chain["population_scale"].min(), max_scale)
ax.plot(lin, density * (1 + 2 * lin ** 2) ** (num_dims / 2), color='k', ls=':',
        label='constant\ndensity 'r'$\left\langle\lambda\right\rangle$')
ax.set_xlabel(r'Population scale $\tau$')
ax.set_ylabel(r'Propensity $\theta$')
ax.legend(loc='best', fontsize='x-small')
ax.text(0.05, 0.95, '(c)', va='top', transform=ax.transAxes)


gs.tight_layout(fig)
fig.savefig('../workspace/addhealth.pdf')
fig.savefig('../workspace/addhealth.png')
print(f'Scale adjustment factor: {scale_factor:.3f}')
```

```{code-cell} ipython3
plt.scatter(*x_samples.T, alpha=0.2)
plt.gca().set_aspect(1)
```

```{code-cell} ipython3
# Get the residuals of latent positions.
x = np.moveaxis(aggregate_best_chain["group_locs"], -1, 0)
x = ys
residuals = x[:, :, None, :] - x[:, None, :, :]
# Estimate the upper and lower bounds of the residuals.
l, u = np.quantile(residuals, [0.025, 0.975], axis=0)
dist_samples = np.square(residuals).sum(axis=-1) ** 0.5
plt.imshow(dist_samples.mean(axis=0))
```

```{code-cell} ipython3
x.shape
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)

for x in [
    np.moveaxis(aggregate_best_chain["group_locs"], -1, 0),
    np.moveaxis(individual_best_chain["group_locs"], -1, 0),
]:
    x = alsm.align_samples(x)
    for i, ax in enumerate(axes.ravel()):
        i = 2 * i
        residuals = x[:, i, :] - x[:, i + 1, :]
        ax.scatter(*residuals.T, alpha=0.1, marker=".")
        # ax.hist(dist_samples[:, i, i + 1])
        print(i, (group_adjacency[i, i + 1] + group_adjacency[i + 1, i]) / group_adjacency[i:i + 2, i:i + 2].sum())
        ax.axvline(0, color="k", ls=":")
        ax.axhline(0, color="k", ls=":")
```

```{code-cell} ipython3
residuals
```
