# Changelog

## Unreleased

- Added `Simulator.with_times(t_span=, save_at=)` and `RebopSimulator.with_times(upto_t=, n_points=)` to store the simulation times on the simulator, like `with_values`, `with_solver` and `with_transform`. Times passed to `solve` still work. Each time argument of `solve` falls back to the stored value independently when not given. `Simulator.interact` takes `t_span` and `save_at` as optional and falls back to the stored times (its `t_span` default is no longer `(0, inf)` but the `solve` default).
- Equations and transforms accept explicit pint quantities and units, such as `x.derive() << -x / (2 * u.min)` or `x.derive() << 3 * u.km / u.hour`, including inside time-dependent parameter defaults. They are replaced by their magnitude in base units when compiling, like the state and parameters. Their units are checked when the `System` is created. Added `units.strip_quantities`.
- fix: scipy solvers (`BDF`, `Radau`, `RK23`, `RK45`, `DOP853`, and `LSODA` through `solve_ivp`) receive a new derivative array on every call instead of the shared output buffer. scipy keeps `f(t, y)` while it estimates the Jacobian by finite differences, so the shared buffer corrupted the estimate: `BDF` and `Radau` failed on stiff systems such as Robertson's, and `RK45` needed about twice as many evaluations.
- fix: the Default column of `variable_table` and `parameter_table` (and so of `model_report` and `notebook_report`) is translated to LaTeX with liblatex. Expression defaults no longer print their internal representation, `transform` and nested paths apply inside them, and pint quantities and units are written in LaTeX (`3\ \frac{1}{\mathrm{s}}`), also inside expressions. Derivative rows show the derivative's own initial value instead of the variable's.
- `notebook_report` (in `poincare.printing`) takes the same `transform`, `descriptions`, `replace_algebraics` and `sections` arguments as `model_report`, so downstream packages can add their own sections. Each section returns Markdown. The default sections are in `default_notebook_sections`.
- Reactions accept a `Parameter` as a constant species: it enters the `MassAction` rate law but has no equation of its own. Supported by `RebopSimulator`, which folds it into the reaction rate.
- Added `n_runs` to `RebopSimulator.solve` to repeat the stochastic simulation with independent random streams and return the average (requires `n_points`).
- fix: `RebopSimulator` passes `MassAction` rates that resolve to a number (parameters with units, products of parameters) to rebop as numbers, so reactions can no longer fire without enough reactant molecules and give negative counts.
- Breaking: reworked fits. `fit_result` returns a `FitResult`, a dict of best values (as before) with the covariance in `.pcov` (`pcov[k, k]` in units of k²). Added `sigma=` (experimental errors in the target's form), `Fitter` (several targets with shared parameters), `fit_steady_state`, and `ToleranceWarning`/`CovarianceWarning`. `make_target` and `Fitter` are exported from `poincare.analysis`. `fit_result` moved to `poincare.analysis.fit_wrappers`. Migration: import it from `poincare.analysis`.
- Added `strip_units` to get a copy of a `System` with every quantity replaced by its magnitude in base units. The removed units are kept in `_units`.
- Breaking: `Simulator` no longer takes `solver=`, which was ignored. Migration: use `with_solver`.
- Breaking: `SteadyState.solve` no longer takes `values=`. Migration: `SteadyState().solve(sim.with_values(values))`.
- `t_span` defaults to `(save_at[0], save_at[-1])`. The `odeint` path of `LSODA` honours an explicit `t_span` that starts before `save_at[0]`.
- Missing initial values or parameter defaults are reported when solving, not when creating the `Simulator`, so they can be supplied with `with_values`.
- `RebopSimulator` has `with_transform`, and accepts variable values with units (the magnitudes are used as copy numbers).
- Only abstract `System`s can be subclassed. Subclassing any concrete `System` raises `TypeError`.
- fix: reports no longer crash when a variable has no equation (for example a constant volume).
- fix: the `numba` backend compiles with symbolite's `libnumba`.
- fix: transform outputs are always floats, instead of taking the integer dtype of the state.

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
