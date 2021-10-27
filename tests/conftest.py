import numpy as np
import os


if seed := os.environ.get('SEED'):
    np.random.seed(int(seed))
