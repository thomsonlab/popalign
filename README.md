# Popalign framework

This is a Python3 implementation of the Popalign framework (this package will not run with Python2).

## Introduction

We develop a software package PopAlign for users to analyze and compare single-cell sequencing data across hundreds of experimental samples. The Popalign framework builds probabilistic gaussian mixture models of single-cell data within a compressed low-dimensional space, and uses statistical metrics to compare the models. Each GMM dissects complex populations into constituent subpopulations, which can then be individually compared. The framework 1) Ranks samples based on shifts across the entire population 2) Align subpopulations across samples based on minimum divergence 3) Finds subpopulation specific transcriptional and abundance changes 4) Automatically discovers differentially expressed genes based on L1 error in the distributions and 5) Queries data using a common universal model.

## Install with pip

Install the package with `pip` in a terminal. This will install the required dependencies.
```sh
pip install git+https://github.com/thomsonlab/popalign.git
```

Update the package with:
```
pip install --upgrade git+https://github.com/thomsonlab/popalign.git
```

Import the package in `python`:
```python
import popalign
```

## Install with git

You can clone this package directly using the `git clone` command in a terminal:
```sh
git clone https://github.com/thomsonlab/popalign.git
```

In the package folder, install the dependencies from the terminal with:
```sh
pip install -r requirements.txt
```
