from matplotlib import pyplot as plt
import numpy as np
import os
import pytest


np.seterr(all='raise', under='ignore')
if seed := os.environ.get('SEED'):
    np.random.seed(int(seed))


TEST_IMAGE_ROOT = os.environ.get('TEST_IMAGE_ROOT', 'test_images')


@pytest.fixture
def figure(request: pytest.FixtureRequest):
    # Create a figure and wait for the test to populate it.
    fig = plt.figure()
    yield fig
    if getattr(fig, 'skip', False):
        return

    # Save the figure.
    fig.tight_layout()
    _, filename = request.node.nodeid.split('::')
    os.makedirs(TEST_IMAGE_ROOT, exist_ok=True)
    fig.savefig(os.path.join(TEST_IMAGE_ROOT, filename))
