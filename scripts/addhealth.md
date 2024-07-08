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
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.linalg import orthogonal_procrustes
import os


mpl.rcParams['figure.dpi'] = 144

DATA_ROOT = Path('../data/addhealth')
SEED = 0
SMOKE_TEST = "CI" in os.environ
```

```{code-cell} ipython3
def pop_if_match(lines: list, pattern: str, index=0) -> re.Match:
    """
    Pop a line from the list if it matches a pattern.
    """
    line = lines[index]
    if (match := re.match(pattern, line)):
        lines.pop(index)
        return match
    else:
        raise ValueError(f'{line} does not match `{pattern}`')


def lines_to_array(lines, parser=float):
    return np.asarray([[parser(cell) for cell in line.split()] for line in lines])


# Load the edgelist.
with open(DATA_ROOT / 'comm72.dat') as fp:
    lines = fp.readlines()

pop_if_match(lines, 'DL')
num_nodes = int(pop_if_match(lines, r'N=(\d+)').group(1))
pop_if_match(lines, 'FORMAT=EDGELIST1')
pop_if_match(lines, 'DATA:')
edgelist = lines_to_array(lines).astype(int)

# Construct the adjacency matrix.
i, j, w = edgelist.T
adjacency = np.zeros((num_nodes, num_nodes), int)
adjacency[i - 1, j - 1] = 1
```

```{code-cell} ipython3
# Load the metadata.
with open(DATA_ROOT / 'comm72_att.dat') as fp:
    lines = fp.readlines()

pop_if_match(lines, 'DL')
num_rows, num_cols = map(int, pop_if_match(lines, r'NR=(\d+), * NC=(\d+)').groups())
assert num_rows == num_nodes
pop_if_match(lines, 'FORMAT = FULLMATRIX DIAGONAL PRESENT')

# Get the column labels.
pop_if_match(lines, 'COLUMN LABELS:')
labels = [label.strip('\n"') for label in lines[:num_cols]]
lines = lines[num_cols:]

# Skip to the data.
while not lines.pop(0).startswith('DATA:'):
    pass

# Create a dataframe for the attributes.
attributes = lines_to_array(lines, parser=int)
attributes = pd.DataFrame(attributes, columns=labels)
assert attributes.shape == (num_nodes, num_cols)
attributes.describe()
```

```{code-cell} ipython3
# Group the adjacency matrix (using the neat `ngroup` function).
keys = ['grade', 'sex']
grouper = attributes.groupby(keys)
group_idx = grouper.ngroup().values
group_sizes = np.bincount(group_idx)
num_groups, = group_sizes.shape
grouping = alsm.evaluate_grouping_matrix(group_idx)
group_adjacency = grouping @ adjacency @ grouping.T

# Get attributes of the groups.
group_attributes = pd.DataFrame([key for key, _ in grouper], columns=keys)

plt.imshow(group_adjacency)
group_attributes
```

```{code-cell} ipython3
# Assemble the data for stan.
num_dims = 2
data = {
    'num_nodes': num_nodes,
    'num_groups': num_groups,
    'num_dims': num_dims,
    'group_idx': group_idx + 1,
    'epsilon': 1e-20,
    'group_adjacency': group_adjacency,
    'adjacency': adjacency,
    'group_sizes': group_sizes,
    'weighted': 0,
}
```

```{code-cell} ipython3
# Fit the model. We will rearrange the groups such that the "furthest" groups are represented by the
# first two indices such that we end up pinning the posterior and killing rotational symmetry.

index = np.arange(num_groups)
index[1] = num_groups - 1
index[num_groups - 1] = 1

stan_file = alsm.write_stanfile(alsm.get_group_model_code())
aggregate_posterior = cmdstanpy.CmdStanModel(stan_file=stan_file)
aggregate_fit = aggregate_posterior.sample(
    iter_warmup=10 if SMOKE_TEST else None,
    iter_sampling=10 if SMOKE_TEST else None,
    chains=3 if SMOKE_TEST else 24,
    inits=1e-2,
    seed=SEED,
    data=alsm.apply_permutation_index(data, index),
    show_progress=False,
)
```

```{code-cell} ipython3
lps = alsm.get_samples(aggregate_fit, 'lp__', False)
plt.plot(lps, alpha=.5)

# Show the number of divergent samples and median lp by chain.
metrics = pd.DataFrame({
    'num_divergent': aggregate_fit.method_variables()['divergent__'].sum(axis=0),
    'median_lp': np.median(lps, axis=0),
})
metrics.sort_values('median_lp')
```

```{code-cell} ipython3
chain = alsm.get_chain(aggregate_fit, metrics.median_lp.argmax())
chain = alsm.apply_permutation_index(chain, alsm.invert_index(index))
print('median leapfrog steps in chain', np.median(chain['n_leapfrog__']))
print('num divergent in chain', np.sum(chain['divergent__']))
```

```{code-cell} ipython3
# Align samples and estimate the mode.
samples = np.rollaxis(chain['group_locs'], -1)
aligned = alsm.align_samples(samples)
modes = alsm.estimate_mode(np.rollaxis(aligned, 1))

fig, ax = plt.subplots()
alsm.plot_edges(modes, group_adjacency, lw=3)

c = group_attributes.grade.values[:, None] * np.ones(aggregate_fit.num_draws_sampling)
pts = ax.scatter(*aligned.T, c=c, marker='.', alpha=.01)
plt.draw()

for xy, radius, tup in zip(modes, np.median(chain['group_scales'], axis=-1),
                           group_attributes.itertuples()):
    color = pts.cmap(pts.norm(tup.grade))
    circle = mpl.patches.Circle(xy, radius, facecolor='none', edgecolor=color)
    ax.add_patch(circle)
    ax.scatter(*xy, color=color, marker='s' if tup.sex == 1 else 'o', zorder=2).set_edgecolor('w')

ax.set_aspect('equal')
```

```{code-cell} ipython3
l, u = np.percentile(chain['ppd_group_adjacency'], [25, 75], axis=-1)
coverage = np.mean((data['group_adjacency'] >= l) & (data['group_adjacency'] <= u))
print(f'ppd coverage of interquartile range: {coverage:.3f}')
```

```{code-cell} ipython3
code = alsm.get_individual_model_code(group_prior=True)
individual_model = cmdstanpy.CmdStanModel(stan_file=alsm.write_stanfile(code))

approximations = []
# Get the centred modes.
y = modes - modes.mean(axis=0)
for seed in range(aggregate_fit.chains):
    approx = individual_model.variational(data, seed=seed + SEED, inits=1e-2)
    approximations.append(approx)
    # Evaluate the aligned loss.
    x = approx.stan_variable('group_locs')
    x = x - x.mean(axis=0)
    transform, _ = orthogonal_procrustes(x, y)
    x = x @ transform
    loss = np.mean(np.square(x - y))
    print(f'seed {seed}; elbo {alsm.get_elbo(approx)}; alignment loss: {loss}')

# Get the best ELBO.
approx = max(approximations, key=alsm.get_elbo)
```

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


fig = plt.figure()
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
plt.setp(ax1.xaxis.get_ticklabels(), visible=False)
ax3 = fig.add_subplot(gs[:, 1])

variational_samples = approx.variational_sample_pd.copy()
variational_samples.columns = approx.column_names

# The individual-level fit is x, the group-level fit is y.
rotation = alsm.evaluate_rotation_matrix(np.deg2rad(-100))
xs = approx.stan_variable('locs') @ rotation
x = approx.stan_variable('group_locs') @ rotation
y = modes

# Center both.
x = x - x.mean(axis=0)
xs = xs - x.mean(axis=0)
y = y - y.mean(axis=0)
ys = aligned

# Scale the group-level fit.
scale_factor = variational_samples.population_scale.mean() / np.mean(chain['population_scale'])
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

for xy, radius, tup in zip(y, np.median(chain['group_scales'], axis=-1) * scale_factor,
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
ax.scatter(chain['population_scale'], chain['propensity'], marker='.', label='group model',
           alpha=.5)
ax.scatter(variational_samples.population_scale, variational_samples.propensity, marker='.',
           label='individual\nmodel', alpha=.5)
# Plot the degeneracy line.
density = group_adjacency.sum() / (num_nodes * (num_nodes - 1))
max_scale = np.sqrt((density ** (- 2 / num_dims) - 1) / 2)
lin = np.linspace(variational_samples.population_scale.min(), max_scale)
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
