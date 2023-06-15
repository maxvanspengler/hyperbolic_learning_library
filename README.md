# Hyperbolic Learning Library

[![Documentation Status](https://readthedocs.org/projects/hyperbolic-learning-library/badge/?version=latest)](https://hyperbolic-learning-library.readthedocs.io/en/latest/?badge=latest)
![Unit Tests](https://github.com/maxvanspengler/hyperbolic_pytorch/workflows/Run%20Unit%20Tests/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort: checked](https://img.shields.io/badge/isort-checked-yellow)](https://github.com/PyCQA/isort)

An extension of the PyTorch library containing various tools for performing deep learning in hyperbolic space. 

Contents:
* [Documentation](#documentation)
* [Installation](#installation)
* [BibTeX](#bibtex)


## Documentation
Visit our [documentation](https://hyperbolic-learning-library.readthedocs.io/en/latest/index.html) for tutorials and more.


## Installation

The Hyperbolic Learning Library was written for Python 3.10+ and PyTorch 1.11+. 

It's recommended to have a
working PyTorch installation before setting up HypLL:

* [PyTorch](https://pytorch.org/get-started/locally/) installation instructions.

Start by setting up a Python [virtual environment](https://docs.python.org/3/library/venv.html):

```
python -venv .env
```

Activate the virtual environment on Linux and MacOs:
```
source .env/bin/activate
```
Or on Windows:
```
.env/Scripts/activate
```

Finally, install HypLL from PyPI.

```
pip install hypll
```

## BibTeX
If you would like to cite this project, please use the following bibtex entry
```
@article{spengler2023hypll,
  title={HypLL: The Hyperbolic Learning Library},
  author={van Spengler, Max and Wirth, Philipp and Mettes, Pascal},
  journal={arXiv preprint arXiv:2306.06154},
  year={2023}
}
```
