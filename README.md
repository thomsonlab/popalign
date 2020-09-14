# Popalign framework

This is a Python3 implementation of the Popalign framework (this package will not run with Python2).

## Introduction

The PopAlign package allows users to analyze and compare single-cell sequencing data across hundreds of experimental samples. The Popalign framework builds probabilistic Gaussian mixture models of single-cell data within a compressed low-dimensional space, and uses statistical metrics to compare the models. Each GMM dissects complex populations into constituent subpopulations, which can then be individually compared. The framework can : 
1) Rank samples based on shifts across the entire population 
2) Align subpopulations across samples based on minimum divergence 
3) Find subpopulation specific transcriptional and abundance changes 
4) Automatically discover differentially expressed genes based on L1 error in the distributions and 
5) Query data using a common universal model

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
import popalign as PA
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

## Tutorial

A Google Colabs notebook tutorial is now available in the 'tutorials/' folder and can also be accessed at the following link: 
https://colab.research.google.com/drive/1t5W_r9FzQCBHe4IL5hCABda5VGAdeTWS?usp=sharing

Full documentation of the PopAlign package is still under construction. Not all features currently implemented have been completely documented or highlighted in the tutorial but will be covered in future versions. 
