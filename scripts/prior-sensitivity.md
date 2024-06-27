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
import itertools
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import alsm

mpl.rcParams["figure.dpi"] = 144
figwidth, figheight = mpl.rcParams["figure.figsize"]
```

```{code-cell} ipython3
results = {}

for prior_type, prior_scale in itertools.product(["cauchy", "exponential", "normal"], [1, 5]):
    with open(f"../workspace/simulation-{prior_type}-{prior_scale}.pkl", "rb") as fp:
        result = pickle.load(fp)
        results[(prior_type, prior_scale)] = result
```

```{code-cell} ipython3
fig, axes = plt.subplots(3, 2, figsize=(figwidth, figwidth), sharex=True, sharey=True)
prior_names = {
    "cauchy": "half-Cauchy",
    "normal": "half-normal",
    "exponential": "exponential",
}


for i, ((prior_type, prior_scale), result) in enumerate(results.items()):
    ax = axes.ravel()[i]
    # Get the correlation between inferred and actual scales.
    chain = result["best_chain"]
    data = result["data"]
    median_group_scales = np.median(chain["group_scales"], axis=-1)
    scale_corrcoef = np.corrcoef(
        median_group_scales,
        data["group_scales"],
    )[0, 1]
    delta = np.mean(median_group_scales - data["group_scales"])
    print(
        prior_type, 
        prior_scale, 
        scale_corrcoef,
        delta,
    )

    # Plot the samples.
    modes = result["modes"]
    samples = result["samples"][::2]
    alsm.plot_edges(modes, data['group_adjacency'] ** .5, zorder=0, ax=ax, lw=3)
    c = np.arange(data['num_groups'])[:, None] * np.ones(samples.shape[0])
    ax.scatter(*samples.T, c=c, cmap='tab10', marker='.', alpha=.05)

    ax.text(
        0.05, 
        0.95, 
        fr"({'abcdef'[i]}) {prior_names[prior_type]}({prior_scale}); "
        fr"$\mathrm{{corr}}\left(\sigma,\hat\sigma\right)={scale_corrcoef:.3f}$", 
        transform=ax.transAxes, 
        fontsize="small", 
        va="top",
    )

    if i % 2 == 0:
        ax.set_ylabel("Embedding $z_2$")

for ax in axes[-1]:
    ax.set_xlabel("Embedding $z_1$")

fig.tight_layout()
fig.savefig("../workspace/prior-sensitivity.pdf")
```
