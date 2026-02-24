![PyVersion](https://img.shields.io/pypi/pyversions/poincare?label=python)
![Package](https://img.shields.io/pypi/v/poincare?label=PyPI)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/poincare)
![License](https://img.shields.io/pypi/l/poincare?label=license)
[![CI](https://github.com/dyscolab/poincare/actions/workflows/ci.yml/badge.svg)](https://github.com/dyscolab/poincare/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/dyscolab/poincare/main.svg)](https://results.pre-commit.ci/latest/github/dyscolab/poincare/main)

# Poincaré: simulation of dynamical systems

Poincaré allows to define and simulate dynamical systems in Python.

## Usage

See [poincare's documentation](https://dyscolab.github.io/poincare/) on the dyscolab page, or go directly to the [intro tutorial](https://marimo.app/github.com/dyscolab/dyscolab-tutorials/blob/main/poincare/getting_started_with_poincare.py).

## Installation

It can be installed from PyPI:

```
pip install -U poincare
```

or conda-forge:

```
conda install -c conda-forge poincare
```

## Development

This project is managed by [pixi](https://pixi.sh).
You can install it for development using:

```sh
git clone https://github.com/dyscolab/poincare
cd poincare
pixi run pre-commit-install  # install pre-commit hooks
```

Pre-commit hooks are used to lint and format the project.

### Testing

Run tests using:

```sh
pixi run test
```
