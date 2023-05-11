Least Square Regression using Random Fourier Features
====================================================================================================

This directory provides examples of regression with random Fourier features.


Installation
----------------------------------------------------------------------------------------------------

See [this document](../../SETUP.md) for more details.

### Docker image (recommended)

```console
$ docker pull tiskw/pytorch:latest
$ cd PATH_TO_THE_ROOT_DIRECTORY_OF_THIS_REPO
$ docker run --rm -it -v `pwd`:/work -w /work -u `id -u`:`id -g` tiskw/pytorch:latest bash
$ cd examples/least_square_regression/
```

### Install on your environment (easier, but pollute your development environment)

```console
$ pip3 install docopt numpy scipy scikit-learn  # Necessary packages
$ pip3 install optuna                           # Required only for hyper parameter tuning
```


Usage
----------------------------------------------------------------------------------------------------

### A simple example of regression

The script file `main_rff_regression_plain.py` provides an example of the simplest usage of
`rfflearn.RFFRegression`. The target function in this script is y = sin(x^2) which is tough for
linear regression to fit well.

```console
$ python3 main_rff_regression_plain.py
```

The following figure shows regression results for the function y = sin(x^2) where the dimension
of RFF is 16.

<div align="center">
  <img src="./figure_least_square_regression.png" width="640" alt="Regression results for function y = sin(x^2) with RFF" />
</div>

### Training on GPU

Open the script file, replace `rfflearn.cpu` as `rfflean.gpu` and run the script again.
