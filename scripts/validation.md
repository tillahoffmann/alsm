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
from scipy.spatial.distance import pdist
import os
import pickle
import networkx as nx
from datetime import datetime


mpl.rcParams["figure.dpi"] = 144
SMOKE_TEST = "CI" in os.environ
SEED = int(os.environ.get("SEED", "7"))
NUM_GROUPS = int(os.environ.get("NUM_GROUPS", "10"))
NUM_DIMS = int(os.environ.get("NUM_DIMS", "2"))
OUTPUT = os.environ.get("OUTPUT", "../workspace/validation-default.pkl")

np.random.seed(SEED)
```

```{code-cell} ipython3
n_connected_tries = 0
while True:
    group_sizes = np.random.poisson(100, NUM_GROUPS)
    num_nodes = group_sizes.sum()
    # group_scales = np.abs(np.random.normal(0, 1, NUM_GROUPS))
    # population_scale = 2.5 * np.abs(np.random.normal(0, 1))
    group_scales = np.random.gamma(3, 1 / 3, NUM_GROUPS)
    population_scale = 2 * np.random.gamma(3, 1 / 3)
    propensity = 10 / num_nodes * (1 + 2 * population_scale ** 2) ** (NUM_DIMS / 2)
    propensity = np.clip(propensity * np.exp(np.random.normal(0, 0.1)), 0, 1)
    
    data = alsm.generate_data(
        group_sizes,
        NUM_DIMS,
        weighted=False,
        group_scales=group_scales,
        population_scale=population_scale,
        propensity=propensity,
    )
    x = data["group_adjacency"] + data["group_adjacency"].T
    symmetric_graph = nx.from_numpy_array(x)
    n_connected_tries += 1
    if nx.is_connected(symmetric_graph):
        break

data["adjacency"].sum(axis=0).mean()
```

```{code-cell} ipython3
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
# Apply a permutation so the two nodes that are furthest apart in the graph are "pinned" 
# down.
inverse = nx.from_numpy_array(np.where(x, 1 / x, 0))
distance, i, j = max([
    (distance, i, j) for i, distances in nx.all_pairs_dijkstra(inverse) 
    for j, distance in distances[0].items()
])

distance, i, j
```

```{code-cell} ipython3
data['epsilon'] = 1e-20

# Fit the model.
model_code = alsm.get_group_model_code(
    scale_prior_type="cauchy",
    scale_prior_scale=1.0,
)
posterior = cmdstanpy.CmdStanModel(stan_file=alsm.write_stanfile(model_code))

start = datetime.now()
fit = posterior.sample(
    data,
    iter_warmup=10 if SMOKE_TEST else None,
    iter_sampling=27 if SMOKE_TEST else None,
    chains=3 if SMOKE_TEST else 8,
    seed=SEED,
    inits=1e-2,
    show_progress=False,
)
print(f"fitted model in {datetime.now() - start}")
```

```{code-cell} ipython3
# Find the best chain.

metrics = []
method_variables = fit.method_variables()
divergent = method_variables["divergent__"].mean(axis=0)
stepsize = method_variables["stepsize__"].mean(axis=0)
for i in range(fit.chains):
    chain = alsm.get_chain(fit, i)
    median_lp = np.median(chain['lp__'])

    # Align the samples.
    samples = np.rollaxis(chain['group_locs'], -1)
    aligned = alsm.align_samples(samples)

    # Information criterion.
    elppd = alsm.util.evaluate_elppd(chain["log_likelihood"])

    print("; ".join([
        f'chain {i}',
        f'median lp: {median_lp:.3f}',
        f'divergent: {divergent[i]:.3f}',
        f'stepsize: {stepsize[i]:.3g}',
        f'elppd: {elppd:.3f}',
    ]))

    # Ignore chains where more than half the samples have diverged.
    metrics.append(-1e9 if divergent[i] > 0.5 or stepsize[i] < 1e-4 else median_lp)

metrics = np.asarray(metrics)
best_chain_idx = np.argmax(metrics)
print(f"best chain: {metrics[best_chain_idx]}")
chain = alsm.get_chain(fit, best_chain_idx)
best_chain_idx
```

```{code-cell} ipython3
# Obtain correlation between scales ...
scale_corr = np.corrcoef(
    np.median(chain["group_scales"], axis=-1),
    group_scales,
)[0, 1]
print("scale correlation", scale_corr)

# ... and distances.
dist_corr = np.corrcoef(
    np.median([pdist(x) for x in np.moveaxis(chain["group_locs"], -1, 0)], axis=0),
    pdist(data["group_locs"]),
)[0, 1]
print("distance correlation", dist_corr)
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
            "best_chain_idx": best_chain_idx,
            "best_chain": chain,
            "data": data,
            "metrics": np.asarray(metrics),
            "samples": samples,
            "fit": fit,
            "scale_corr": scale_corr,
            "dist_corr": dist_corr,
            "n_connected_tries": n_connected_tries,
        }, fp)
```
