# Latent Space Approaches to Aggregate Network Data [![](https://github.com/tillahoffmann/alsm/actions/workflows/main.yml/badge.svg)](https://github.com/tillahoffmann/alsm/actions/workflows/main.yml)

This repository contains the notebooks and Stan and Python code required to reproduce the results in the accompanying manuscript "[Latent Space Approaches to Aggregate Network Data](https://arxiv.org/abs/2303.08338)". From the abstract,

> Large-scale network data can pose computational challenges, be expensive to acquire, and compromise the privacy of individuals in social networks. We show that the locations and scales of latent space cluster models can be inferred from the number of connections between groups alone. We demonstrate this modelling approach using synthetic data and apply it to friendships between students collected as part of the Add Health study, eliminating the need for node-level connection data. The method thus protects the privacy of individuals and simplifies data sharing. It also offers performance advantages over node-level latent space models because the computational cost scales with the number of clusters rather than the number of nodes.

## Reproducing the Results

Reproducing the results is straightforward by following these steps.

1. Set up a clean Python environment. This code has been tested with Python 3.10 on macOS and Ubuntu.
2. Install the Python dependencies by running `pip install -r requirements.txt` from the root directory of this repository.
3. Install `cmdstan`, the command line interface to the probabilistic programming framework [Stan](https://mc-stan.org), by running `python -m cmdstanpy.install_cmdstan --version=2.34.0`; this may take a few minutes depending on your machine. Other recent versions of Stan may also be compatible but have not been tested.
4. Optionally, run `make tests` to test the installation and runtime environment.
5. Run `make data` to download the Adolescent to Adult Health network data.
6. Run `make analysis` to run all analysis. The results will be saved in a new `workspace` folder at the root of the repository. Results comprise `.html` files summarizing the analysis and `.pdf` and `.png` files for the figures in the manuscript.

The source code comprises two parts: first, the Python package `alsm` (containing the Stan model code and utility functions) and, second, Jupyter notebooks stored as `.md` [jupytext](https://jupytext.readthedocs.io/) files in the `scripts` folder (containing the code to run analysis and produce figures). If you are familiar with jupytext, go right ahead and open the `.md` files as a notebook. If you prefer traditional `.ipynb` files, run `make ipynb` to generate `.ipynb` notebooks which will be stored in the `scripts` folder.
