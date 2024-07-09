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
import pathlib
import numpy as np
```

```{code-cell} ipython3
filenames = list(pathlib.Path("../workspace/").glob("validation-*.pkl"))
stats = {}
for filename in filenames:
    if "default" in filename.name:
        continue
    with open(filename, "rb") as fp:
        result = pickle.load(fp)
    stats.setdefault("scale_corr", []).append(result["scale_corr"])
    stats.setdefault("dist_corr", []).append(result["dist_corr"])
len(filenames)
```

```{code-cell} ipython3
for key, corr in stats.items():
    print(f"{key} {np.median(corr):.3f}")
```
