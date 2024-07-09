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
import pickle
import gc


mpl.rcParams['figure.dpi'] = 144

DATA_ROOT = Path('../data/addhealth')
SEED = 0
SMOKE_TEST = "CI" in os.environ
```

# Load data

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

# Group-level fit

```{code-cell} ipython3
# Fit the model. We will rearrange the groups such that the "furthest" groups are represented by the
# first two indices such that we end up pinning the posterior and killing rotational symmetry.

index = np.arange(num_groups)
index[1] = num_groups - 1
index[num_groups - 1] = 1

stan_file = alsm.write_stanfile(alsm.get_group_model_code())
aggregate_model = cmdstanpy.CmdStanModel(stan_file=stan_file)
aggregate_fit = aggregate_model.sample(
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
with open("../workspace/addhealth-aggregate.pkl", "wb") as fp:
    pickle.dump({
        "data": data,
        "fit": aggregate_fit,
        "attributes": attributes,
        "group_attributes": group_attributes,
        "permutation_index": index,
    }, fp)
```

# Individual level fit

```{code-cell} ipython3
stan_file = alsm.write_stanfile(alsm.get_individual_model_code(group_prior=True))
individual_model = cmdstanpy.CmdStanModel(stan_file=stan_file)

num_batches = 4
batch_size = 6

for i in range(num_batches):
    individual_fit = individual_model.sample(
        iter_warmup=10 if SMOKE_TEST else None,
        iter_sampling=10 if SMOKE_TEST else None,
        chains=3 if SMOKE_TEST else batch_size,
        inits=1e-2,
        seed=SEED + i * batch_size,
        data=data,
        show_progress=True,
    )

    with open(f"../workspace/addhealth-individual-{i}.pkl", "wb") as fp:
        pickle.dump({
            "data": data,
            "fit": individual_fit,
        }, fp)

    # Try to free up some memory so we can load the next batch without issues.
    del individual_fit
    print("collected", gc.collect())
```

```{code-cell} ipython3

```
