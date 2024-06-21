---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import cmdstanpy
from matplotlib import pyplot as plt
import numpy as np
```

```{code-cell} ipython3
%%writefile mode-separation-demo.stan

// Demonstration that Stan can't explore modes that are well-separated.

data {
    real loc, scale;
}

parameters {
    real x;
}

model {
    target += log((exp(normal_lpdf(x | 0, scale)) + exp(normal_lpdf(x | loc, scale))) / 2);
}
```

```{code-cell} ipython3
model = cmdstanpy.CmdStanModel(stan_file="mode-separation-demo.stan")
loc = 10
scale = 1
data = {"loc": loc, "scale": scale}
fit = model.sample(data, chains=1, iter_sampling=10_000, iter_warmup=10_000, seed=0)

lin = np.linspace(- 4 * scale, loc + 4 * scale, 200)
lpdf = [model.log_prob({"x": x}, data).lp__[0] for x in lin]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.hist(fit.x, density=True, bins=21, color="silver", label="Stan samples")
ax.plot(lin, np.exp(lpdf), label="target density")
ax.set_xlabel("Parameter $x$")
ax.set_ylabel("Density $p(x)$")
ax.legend()
fig.savefig("mode-separation-demo.pdf")
```
