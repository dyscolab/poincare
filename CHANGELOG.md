# Changelog

## 1.3.0

- Breaking: reworked fits. `fit_result` returns a `FitResult` (dict of best values with the covariance in `.pcov`) and is exported from `poincare.analysis`. Added `sigma=`, `Fitter` (several targets with shared parameters), `fit_steady_state`, `ToleranceWarning` and `CovarianceWarning`.
- Breaking: removed the ignored `Simulator(solver=)` (use `with_solver`) and `SteadyState.solve(values=)` (use `sim.with_values(values)`).
- Added `Simulator.with_times` and `RebopSimulator.with_times` to store simulation times on the simulator. `solve` and `interact` fall back to the stored times. `t_span` defaults to `(save_at[0], save_at[-1])`.
- Equations and transforms accept explicit pint quantities and units. They are checked when the `System` is created and converted to base units when compiling. Added `units.strip_quantities` and `strip_units`.
- `RebopSimulator`: added `n_runs` (averaged repeated runs), `with_transform`, `Parameter`s as constant species, and dimensionless initial values such as `u.count`. Non-dimensionless initial values raise `DimensionalityError`.
- `notebook_report` takes the same arguments as `model_report`, plus custom `sections`.
- Bundled the `dyscolab-modeling` agent skill, with `python -m poincare.skill install` and `check`.
- Missing initial values or parameter defaults are reported when solving, so they can be supplied with `with_values`.
- Only abstract `System`s can be subclassed.
- fix: scipy solvers no longer share the derivative buffer, which corrupted finite-difference Jacobians (`BDF` and `Radau` failed on stiff systems).
- fix: `RebopSimulator` passes numeric `MassAction` rates as numbers, preventing negative counts.
- fix: reports render defaults in LaTeX (expressions, quantities, derivative initial values), list parameters that depend on variables or time, and no longer crash on variables without an equation.
- fix: the `numba` backend compiles with symbolite's `libnumba`.
- fix: transform outputs are always floats.

## 1.2.0

- Breaxing: Simulation time (`save_at` and `upto_t`) dimensionality is now enforced with errors instead of warnings.
- Breaking: Simulation times are now set on simulator with `with_values` instead of at simualtion time, solvers an transform get a similar API. All simulation functions now take simulators as inputs.
- Added `fit_result` to `analysis`.
- fix: normalize_eq in `printing.latex` can now handle equations which compile to `str` instead of `LatexExpr`.

## 1.1.2

- Independent variable units are now inferred from equations and derivatives.
- Units in save_at are now handled properly, raises `FutureWarning` if they don't match the implicit units.

## 1.1.1

- Fix: transform now handles events properly.
- Variables and parameters belonging to nested systems are now printed in latex with full path.

## 1.1.0

- Added `model_report()` to generate reports with model's equations, variables and parameters in LaTeX.
- Added support for stochastic simulations with rebop in reactions.
- Fix: `RateLaw` now use their parents reactant when copied if available.
- Update symbolite dependency 1 to 1.1.0.

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
