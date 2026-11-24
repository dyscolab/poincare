# Public API inventory: poincare (1.2.0 and newer)

Use this to check that a name exists in these versions before you write code with it. When
the installed version differs, check the installed source instead
(`python -c "import poincare.analysis as m; print(dir(m))"`).

Single-file modules (for example `poincare.solvers` or `poincare.simulator`) are part of
the public API just like package `__init__` exports.

Names marked *(newer)* exist only after poincare 1.2.0.

The simbio and jablonski inventories are in [simbio-api.md](simbio-api.md) and
[jablonski-api.md](jablonski-api.md). Those files come with the simbio and jablonski
packages, so they are only here when the library is installed.

## poincare

| Module | Names | Reference |
| --- | --- | --- |
| `poincare` | `System`, `Variable`, `Derivative`, `Parameter`, `Constant`, `Independent`, `initial`, `assign`, `Simulator`, `SteadyState`, `Oscillations`, `model_report`, `strip_units` *(newer)* | [poincare.md](poincare.md) |
| `poincare.solvers` | `LSODA`, `RK23`, `RK45`, `DOP853`, `Radau`, `BDF` (and the `Solver` protocol) | [poincare.md](poincare.md#simulator) |
| `poincare.simulator` | `Simulator`, plus the `Components` type alias (Constant | Parameter | Variable | Derivative) | [poincare.md](poincare.md#simulator) |
| `poincare.analysis` | `SteadyState`, `Oscillations`, `Sweeper`, `fit_result`; *(newer)* `Fitter`, `fit_steady_state`, `make_target` | [analysis.md](analysis.md) |
| `poincare.analysis.fits` | `fit_result` (1.2.0 only; newer in `poincare.analysis.fit_wrappers`), `make_target`; *(newer)* `Fitter`, `FitResult`, `Covariance`, `ToleranceWarning`, `CovarianceWarning` | [analysis.md](analysis.md#fitting-parameters-to-data) |
| `poincare.units` *(newer)* | `strip_units` | [units.md](units.md) |
| `poincare.reactions` | `Reactant`, `reaction_initial`, `RateLaw`, `MassAction`, `AbsoluteRateLaw` | [poincare.md](poincare.md#reaction-primitives) |
| `poincare.reactions.rebop` | `RebopSimulator` | [analysis.md](analysis.md#stochastic-simulation-rebop) |
| `poincare.printing.latex` | `model_report`, `latex_equations`, `parameter_table`, `variable_table`, `ToLatex` | [poincare.md](poincare.md#reports-and-introspection) |
| `poincare.printing` (newer than 1.2.0) | `notebook_report`, `default_notebook_sections`, `model_report`, `latex_equations`, `parameter_table`, `variable_table` | [poincare.md](poincare.md#reports-and-introspection) |
| `poincare.helpers` | `get_from`, `to_values` | [poincare.md](poincare.md#reports-and-introspection) |
| `poincare.skill` *(newer)* | `install`, `check` (also `python -m poincare.skill install` / `check`): install and check this skill | [SKILL.md](../SKILL.md#versions-and-environment) |

## Internal helpers (don't use)

These are importable but are implementation details, or exist only to support the
functions above: `poincare.analysis.fits.UnitsHandler` / `parse_p0`, `poincare.compile.*`
(compiler internals), `poincare.simulator.rescale` / `get_scale` / `Problem`, the
`poincare.printing.latex` formatting helpers not listed above, and anything whose name
starts with `_`.

## Not available in 1.2.0

Don't look for these in poincare 1.2.0. Use the documented alternative instead. The third
column says which newer versions add them.

| Missing in 1.2.0 | Use instead | Newer versions |
| --- | --- | --- |
| A `Fitter` class, `fit_steady_state`, fit covariances | `fit_result` (best values only) | `Fitter`, `fit_steady_state`, `FitResult.pcov` |
| `poincare.strip_units` | Build the model without units, or dequantify results | `poincare.strip_units` |
| `n_runs` argument on rebop solvers | A loop over seeds + `xr.concat(...).mean("run")` | `n_runs=` on `RebopSimulator.solve` (and the jablonski rebop functions) |
| `Simulator.with_times`, `RebopSimulator.with_times` | Pass the times to `solve` (`save_at=`, `t_span=`; rebop: `upto_t=`, `n_points=`) | `with_times` (recommended) |
| `RebopSimulator.with_transform` | Compute derived outputs from the result | `with_transform` |
| A `Parameter` as a reactant (constant species) | Fold the constant into the rate (`rate=k * G_value`) | `Parameter` reactants |
