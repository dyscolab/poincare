# Changelog

## 1.1.0

- Added `model_report()` to generate reports with model's equations, variables and parameters in LaTeX.
- Added support for stochastic simulations with rebop in reactions.
- Fix: `RateLaw` now use their parents reactant when copied if available.
- Update symbolite dependency 1o 1.1.0.

## 1.0.0

- Major changes for release 1.0.0.

## 0.6.3

- Fix `Simulator.interact` when default values are negative (@tdinapoli).

## 0.6.2

- Add upper bound for `symbolite<0.7`.

## 0.6.1

- Update to `scipy-events >= 0.3.1`.

## 0.6.0

- Add event handling to `Simulator` using `scipy-events`.
- Change `System.variables` and `System.parameters` from `DataFrame` to `Series`.

## 0.5.0

- Update to `symbolite >= 0.6` to support serialization (pickle) of `System`.
- Add `__getnewargs__` to `Variable` and `Derivative` to support serialization
  of "locally-defined" `System`s with `cloudpickle`.

## 0.4.0

- `System.variables` and `System.paramters` return a `pandas.DataFrame` instead of a `Table` object.

## 0.3.1

- Add `__signature__` to `System`, for environments that do not support `dataclass_transform`.

## 0.3.0

- Rename `times` to `save_at` in `Simulator.interact`.
- Fix typing in `Simulator.solve` to accept `ArrayLike`.
- Improve `__repr__` for `System` type to show the number of components.
- Change `Backend` from `Enum` to `Literal`. No need to import the enum from `.compile`.

## 0.2.0

- Allow different ODE solvers and implement wrappers for SciPy and numbalsoda.
- Rename `Time` to `Independent` and `times` to `save_at` in `Simulator.solve`.
  Allows custom names for the independent variable
  and adding more then one independent variable in the future.
- Allow Sequence of variables in `Simulator.transform`.
- Several improvements to `Simulator.interact`.
- Add `ipywidgets` to extra requirements.
- Several improvements in typing and error messages.
- Prevent subclassing concrete systems (for now).

## 0.1.0

- First release of Poincaré.
