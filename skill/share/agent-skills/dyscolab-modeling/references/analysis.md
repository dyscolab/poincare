# Analysis with poincare 1.2: steady states, sweeps, oscillations, fits, stochastic runs

All of these take a `Simulator` (configure values, solver and transforms on it first).
For simbio and jablonski models, import them from `poincare`, because simbio and jablonski
do not re-export them.

## Contents

- [Steady states](#steady-states)
- [Parameter sweeps](#parameter-sweeps)
- [Bistability](#bistability)
- [Oscillations and limit cycles](#oscillations-and-limit-cycles)
- [Fitting parameters to data](#fitting-parameters-to-data)
- [Stochastic simulation (rebop)](#stochastic-simulation-rebop)

## Steady states

```python
from poincare import SteadyState, Simulator, solvers

sim = Simulator(Model).with_solver(solvers.LSODA(rtol=1e-8, atol=1e-10))
ss = SteadyState().solve(sim.with_values({Model.k: 2}))  # or just solve(sim)
ss["x"].item(), ss["event"].item()  # event == 0 → converged
```

- Pass parameter values through `sim.with_values(...)`. `SteadyState.solve` also accepts
  `values=` in 1.2.0, but newer poincare removes it (`TypeError`), so the `with_values`
  form is the portable one.
- `SteadyState(condition=SmallDerivatives(), t_end=<float max>)` integrates until every
  derivative is small, then stops. The result has one time point plus an `event` column.
- **Accuracy follows the solver tolerances.** The default condition stops when
  `|dy/dt| < 10·atol` or `< 10·rtol·|y|`, so the default solver (rtol 1e-3) gives
  steady states that are only accurate to about 1 %. Use `with_solver(LSODA(rtol=1e-8, atol=...))`.
  Choose `atol` well below the smallest population or concentration you care about.
- Set `t_end` (in base units) to bound runs that might not converge. **In 1.2.0, a run
  that reaches `t_end` without converging raises an obscure
  `ValueError: could not broadcast input array ...`** instead of returning `event != 0`.
  This happens for oscillating or slowly drifting systems. Catch it, then simulate
  a time course or use `Oscillations`. An `atol` too small for the solver to reach the
  convergence condition fails the same way, or with
  `IndexError: index 0 is out of bounds` (newer versions). Loosen `atol` slightly.
- A steady state from integration depends on the initial condition (multistability).
  Set the initial values with `with_values` to pick a branch.

## Parameter sweeps

**Default pattern: loop + `xr.concat`.** It works with units, transforms and any output.

```python
import xarray as xr

values = [0.5, 1, 2] * u.mM / u.s
sweep = xr.concat(
    [SteadyState().solve(sim.with_values({Model.p: v})).isel(time=-1) for v in values],
    dim="p",
).assign_coords(p=values.magnitude)
```

`isel(time=-1)` matters: without it, the runs (which stop at different times) are aligned
on `time` and padded with NaN. The same loop works for time courses
(`sim.with_values({...}).with_times(save_at=...).solve()`, concatenated on a new dimension) and for
2D sweeps (nest two loops, or concat a list of 1D sweeps).

Built-in helpers, which work only in simple cases in 1.2.0:

- `SteadyState().sweep(sim, variable=Model.p, values=[...])` returns the steady state of
  each state variable along a dimension named after the parameter.
  - It **fails when state variables carry units**.
  - It **fails when the simulator has a transform without `append=True`**.
  - It drops transform outputs.
  - Use the loop in those cases.
- `Sweeper(func).sweep(sim, parameter=Model.p, values=[...], save_at=...)` runs a full
  simulation per value and applies `func(dataset)`. Use it for derived scalars such as
  peak height or final value. If `func` returns a dict, each key becomes a variable.

## Bistability

```python
uad = SteadyState().sweep_up_and_down(
    sim, variable=Model.c, values=np.linspace(-3, 3, 30)
)
uad.sel(direction="up")["x"], uad.sel(direction="down")["x"]
SteadyState().bistability(
    sim, variable=Model.c, values=..., atol=0.1, rtol=0.01
)  # bool per value
```

The up/down sweep seeds each run with the previous steady state, so hysteresis shows up as
different branches.

## Oscillations and limit cycles

```python
from poincare import Oscillations

res = Oscillations().sweep(
    sim,
    parameter=Model.w,
    values=np.linspace(1, 2, 20),
    T_min=1.5,
    T_max=20,  # expected period range
    rel_time=100,  # time to discard before the system reaches the limit cycle
    T_after_rel=10,
    timesteps_in_T=10,
    variables=None,  # optional; None = all variables
)
(
    res.sel(quantity="period"),
    res.sel(quantity="amplitude"),
    res.sel(quantity="difference_rms"),
)
```

- The time step is `T_min / timesteps_in_T`, so the period resolution is about that step.
  Increase `timesteps_in_T` for precision.
- **A single run at the current parameters** is a sweep over one value:
  `values=[Model.p.default]` (any parameter works as `parameter`).
- Read results per variable: `res["x"].sel(quantity="period").item()`. On the Dataset,
  `.values` is the dict method, not the data.
- Warnings mean that no period was found or confirmed, and **the returned value can still
  be off** (it falls back to the strongest spectral period). Try a longer `rel_time` or more
  `T_after_rel` periods.
- `difference_rms / amplitude` close to 0 indicates a clean limit cycle.

## Fitting parameters to data

Use this only when the user provides data to fit.

```python
from poincare.analysis import fit_result
from poincare.analysis.fits import make_target

target = make_target(
    {Model.x: x_data, "E": e_data}, save_at=t_data
)  # quantities allowed
fit = fit_result(
    sim,  # may include transforms; target keys must match outputs
    target,
    p0={
        Model.k: (0.5, 0, 10),  # (initial guess, lower, upper)
        Model.x: 1.5,
    },  # initial values can be fitted too
    scale={"E": 0.5},  # optional per-output residual weight
)
fit[Model.k]
```

- In 1.2.0, `fit_result` returns a dict `{component: best value}` and nothing else (no
  covariance). It calls `scipy.optimize.least_squares` (extra `**kwargs` are passed through).
- With units, give the data, `save_at` and `p0` as quantities. The result comes back as
  quantities.
- Use tight solver tolerances. Finite-difference Jacobians are noisy with the default 1e-3.
- If the data are on a different time grid than you want to simulate, `make_target` still
  defines the time points of the residual.

**Newer than 1.2.0 (reworked fits):**

- `fit_result(sim, target, p0=, scale=, sigma=None, **kwargs)` returns a `FitResult`. It is
  still a dict `{component: best value}`, so the code above works unchanged, and adds the
  covariance in `.pcov`:

  ```python
  fit[Model.k]  # best value (quantity if the model has units)
  fit.pcov[Model.k, Model.k] ** 0.5  # standard error, in the units of k
  np.asarray(fit.pcov)  # plain matrix, ordered as fit.pcov.parameters
  ```

  - `sigma=` takes the experimental standard deviations in the same form as the target
    (for example `make_target({Model.x: err * u.mM}, save_at=t)`). Residuals are divided by
    it and the covariance treats it as absolute. Without `sigma`, the noise is estimated
    from the residuals, as in `scipy.optimize.curve_fit`.
  - Warnings: `ToleranceWarning` (solver `rtol`/`atol` too loose for finite differences;
    tighten them or pass `diff_step=1e-4`) and `CovarianceWarning` (a parameter ended at a
    bound, or parameters the data cannot separate). Both are in `poincare.analysis.fits`.

- `make_target` is also exported from `poincare.analysis`.

- `fit_steady_state(sim, SteadyState(), Model.p, target, p0=...)` fits steady states. The
  target's coordinate holds the values of `Model.p` that the data were measured at, as in
  `xr.Dataset({"x": ("p", x_ss)}, coords={"p": p_values})`. It runs `SteadyState().sweep`
  internally, so the sweep's limitations apply (no units on state variables).

- `Fitter` fits several datasets at once with shared parameters. Use one simulation
  function per target:

  ```python
  from poincare.analysis import Fitter, make_target

  fitter = Fitter(
      sim,
      [
          lambda s: s.with_times(save_at=t1).solve(),
          lambda s: s.with_values({Model.x: 2.0}).with_times(save_at=t2).solve(),
      ],  # second experiment
      [make_target({Model.x: x1}, t1), make_target({Model.x: x2}, t2)],
      p0={Model.k: (0.5, 0, 5)},
      sigma=None,  # or one sigma per target
  )
  fit = fitter.fit()  # fit(p0=...) overrides guesses or bounds for one run
  ```

  A simulation function may return a Dataset (compared variable by variable) or a
  DataArray. `scales=` is a number, a `{variable: weight}` mapping, or one of those per
  target.

- jablonski wraps this for spectra and time-resolved emission in `jablonski.fits`
  ([jablonski.md](jablonski.md#fits)).

## Stochastic simulation (rebop)

Gillespie simulation of reaction networks (`MassAction` and `RateLaw` only) via the
`rebop` package (`pip install rebop`).

```python
from poincare.reactions.rebop import (
    RebopSimulator,
)  # also simbio.rebop, jablonski.rebop

rsim = (
    RebopSimulator(Model)
    .with_values({Model.A: 100})
    .with_times(upto_t=10, n_points=100)
)
ds = rsim.solve(rng=1)  # n_points equal intervals → n_points + 1 rows
# poincare 1.2.0 has no with_times: rsim.solve(upto_t=10, n_points=100, rng=1)
```

- Variables are integer copy numbers. In 1.2.0, **variable values may not carry units**
  (`TypeError`). Rate parameters and `upto_t` may. Models with plain `Equation`s
  (`x.derive() << ...`) are rejected.
- Rates may use `+ - * / **`, `real.exp`, `real.sqrt`, `real.hypot`, `real.degrees`,
  `real.radians` and the constants `real.e`, `real.pi`, `real.tau`.
- Average many runs (`xr.concat([... for seed in range(n)], dim="run").mean("run")`) before
  comparing with the ODE solution.
- **Newer than 1.2.0:**
  - `rsim.with_times(upto_t=, n_points=100).solve(rng=1, n_runs=50)` returns the average of 50 runs
    with independent random streams. `n_runs > 1` requires `n_points`.
  - `RebopSimulator.with_times(upto_t=, n_points=None)` (recommended) stores the times so
    `rsim.solve(rng=1)` can be called without them. `solve(upto_t=, n_points=)` still
    works, and each argument not passed to `solve` falls back to its stored value
    independently.
  - `RebopSimulator.with_transform(...)` works as on `Simulator` (use `append=True` to
    keep the copy numbers).
  - Variable values may use dimensionless units such as `u.count` (`20 * u.count` is 20
    copies). Amounts of substance (`u.molecule`, `u.mol`) raise `DimensionalityError`.
    Releases up to 1.2.0 accepted them silently but converted them to moles, so
    `1000 * u.molecule` became 0 copies. Keep stochastic models unitless or use `u.count`.
  - Constant species (`Parameter` reactants, see
    [poincare.md](poincare.md#reaction-primitives)) are folded into the reaction rate.
  - `MassAction` rates that resolve to a number (parameters with units, products of
    parameters) are passed to rebop as numbers. The poincare changelog reports that in
    1.2.0 some of these rates could fire without enough reactant molecules. If a 1.2.0 run
    shows negative counts, this is the cause.
- For jablonski models with time-dependent excitation, use `jablonski.rebop.rebop_piecewise`
  (see [jablonski.md](jablonski.md#stochastic-runs)).
