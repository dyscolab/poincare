![PyVersion](https://img.shields.io/pypi/pyversions/poincare?label=python)
![Package](https://img.shields.io/pypi/v/poincare?label=PyPI)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/poincare)
![License](https://img.shields.io/pypi/l/poincare?label=license)
[![CI](https://github.com/dyscolab/poincare/actions/workflows/ci.yml/badge.svg)](https://github.com/dyscolab/poincare/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/dyscolab/poincare/main.svg)](https://results.pre-commit.ci/latest/github/dyscolab/poincare/main)

# Poincaré: simulation of dynamical systems

Poincaré allows to define and simulate dynamical systems in Python.

### Definition

To define the system

$$ \\frac{dx}{dt} = -x \\quad \\text{with} \\quad x(0) = 1 $$

we can write:

```python
>>> from poincare import Variable, System, initial
>>> class Model(System):
...   # Define a variable with name `x` with an initial value (t=0) of `1``.
...   x: Variable = initial(default=1)
...   # The rate of change of `x` (i.e. velocity) is assigned (<<) to `-x`.
...   # This relation is assigned to a Python variable (`eq`)
...   eq = x.derive() << -x
...
```

### Simulation

To simulate that system:

```python
>>> from poincare import Simulator
>>> sim = Simulator(Model)
>>> sim.solve(save_at=range(3))
             x
time
0     1.000000
1     0.368139
2     0.135501
```

The output is a `pandas.DataFrame`,
which can be plotted with `.plot()`.

### Changing initial conditions

To change the initial condition,
we can pass a dictionary to the `solve` method:

```python
>>> sim.solve(values={Model.x: 2}, save_at=range(3))
             x
time
0     2.000000
1     0.736278
2     0.271002
```

### Transforming the output

We can compute transformations of the output
by passing a dictionary of expressions:

```python
>>> Simulator(Model, transform={"x": Model.x, "2x": 2 * Model.x}).solve(save_at=range(3))
             x        2x
time
0     1.000000  2.000000
1     0.368139  0.736278
2     0.135501  0.271002
```

### Higher-order systems

To define a higher-order system,
we have to assign an initial condition to the derivative of a variable:

```python
>>> from poincare import Derivative
>>> class Oscillator(System):
...   x: Variable = initial(default=1)
...   v: Derivative = x.derive(initial=0)
...   eq = v.derive() << -x
...
>>> Simulator(Oscillator).solve(save_at=range(3))
             x         v
time
0     1.000000  0.000000
1     0.540366 -0.841561
2    -0.416308 -0.909791
```

### Non-autonomous systems

To use the independent variable,
we create an instance of `Independent`:

```python
>>> from poincare import Independent
>>> class NonAutonomous(System):
...   time = Independent()
...   x: Variable = initial(default=0)
...   eq = x.derive() << 2 * time
...
>>> Simulator(NonAutonomous).solve(save_at=range(3))
             x
time
0     0.000000
1     1.000001
2     4.000001
```

### Constants, Parameters, and functions

Besides variables,
we can define parameters and constants,
and use functions from [symbolite](https://github.com/dyscolab/symbolite).

#### Constants

Constants allow to define common initial conditions for Variables and Derivatives:

```python
>>> from poincare import assign, Constant
>>> class Model(System):
...     c: Constant = assign(default=1, constant=True)
...     x: Variable = initial(default=c)
...     y: Variable = initial(default=2 * c)
...     eq_x = x.derive() << -x
...     eq_y = y.derive() << -y
...
>>> Simulator(Model).solve(save_at=range(3))
             x         y
time
0     1.000000  2.000000
1     0.368139  0.736278
2     0.135501  0.271002
```

Now, we can vary their initial conditions jointly:

```python
>>> Simulator(Model).solve(values={Model.c: 2}, save_at=range(3))
             x         y
time
0     2.000000  4.000000
1     0.736278  1.472556
2     0.271001  0.542003
```

But we can break that connection by passing `y`'s initial value directly:

```python
>>> Simulator(Model).solve(values={Model.c: 2, Model.y: 2}, save_at=range(3))
             x         y
time
0     2.000000  2.000000
1     0.736278  0.736278
2     0.271002  0.271002
```

#### Parameters

Parameters are like Variables,
but their time evolution is given directly as a function of time,
Variables, Constants and other Parameters:

```python
>>> from poincare import Parameter
>>> class Model(System):
...     p: Parameter = assign(default=1)
...     x: Variable = initial(default=1)
...     eq = x.derive() << -p * x
...
>>> Simulator(Model).solve(save_at=range(3))
             x
time
0     1.000000
1     0.368139
2     0.135501
```

#### Functions

Symbolite functions are accessible from the `symbolite.scalar` module:

```python
>>> from symbolite import scalar
>>> class Model(System):
...     x: Variable = initial(default=1)
...     eq = x.derive() << scalar.sin(x)
...
>>> Simulator(Model).solve(save_at=range(3))
             x
time
0     1.000000
1     1.951464
2     2.654572
```

### Units

poincaré also supports functions through
[`pint`](https://github.com/hgrecco/pint)
and [`pint-pandas`](https://github.com/hgrecco/pint-pandas).

```python
>>> import pint
>>> unit = pint.get_application_registry()
>>> class Model(System):
...     x: Variable = initial(default=1 * unit.m)
...     v: Derivative = x.derive(initial=0 * unit.m/unit.s)
...     w: Parameter = assign(default=1 * unit.Hz)
...     eq = v.derive() << -w**2 * x
...
>>> result = Simulator(Model).solve(save_at=range(3))
```

The columns have units of `m` and `m/s`, respectively.
`pint` raises a `DimensionalityError` if we try to add them:

```python
>>> result["x"] + result["v"]
Traceback (most recent call last):
...
pint.errors.DimensionalityError: Cannot convert from 'meter' ([length]) to 'meter / second' ([length] / [time])
```

We can remove the units and set them as string metadata with:

```python
>>> result.pint.dequantify()
             x              v
unit     meter meter / second
time
0          1.0            0.0
1     0.540366      -0.841561
2    -0.416308      -0.909791
```

which allows to plot the DataFrame with `.plot()`.

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
