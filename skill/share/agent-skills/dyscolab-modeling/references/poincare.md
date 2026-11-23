# poincare 1.2: declaring and simulating ODE systems

poincare is the base library. simbio and jablonski models are poincare `System`s,
so everything here applies to them too. Notes marked **Newer than 1.2.0** describe poincare
`main`, which is not on PyPI yet.

## Contents

- [Components](#components)
- [Equations](#equations)
- [Parameters, constants and functions](#parameters-constants-and-functions)
- [Composition (subsystems)](#composition-subsystems)
- [Simulator](#simulator)
- [Results](#results)
- [Reaction primitives](#reaction-primitives)
- [Reports and introspection](#reports-and-introspection)

## Components

```python
from poincare import (Constant, Derivative, Independent, Parameter, System, Variable,
                      assign, initial)

class Model(System):
    t = Independent()                               # time; only needed to use t in expressions
    x: Variable = initial(default=1)                # state variable, initial value
    v: Derivative = x.derive(initial=0)             # dx/dt as extra state (2nd-order systems)
    k: Parameter = assign(default=2)                # parameter (number, quantity or expression)
    c: Constant = assign(default=1, constant=True)  # constant; can set initial values
```

| Factory | Creates | Notes |
| --- | --- | --- |
| `initial(default=...)` | `Variable` | Default can be a number, a pint quantity or a `Constant` expression |
| `x.derive(initial=...)` | `Derivative` | Declares a higher-order state. Without `initial`, it is only used in equations |
| `assign(default=...)` | `Parameter` | Default can be an expression of other parameters, variables or time |
| `assign(default=..., constant=True)` | `Constant` | Use it for shared initial conditions (`initial(default=2 * c)`) |
| `Independent(default=0)` | time | One per system. `SpectroscopicSystem` already defines `time` in seconds |

**Annotate every component** (`x: Variable = ...`). The annotations build the keyword-only
constructor used for composition and `with_values`. An annotation that doesn't match the
component type raises `TypeError` when the class is created. Components with no default
(`assign()`) are required constructor arguments. **Newer than 1.2.0:** `Simulator(Model)`
also accepts a model with missing values. `solve` then raises
`TypeError: Missing initial value for k`, unless you supply it with `with_values`.

**Don't subclass a concrete model to extend it.** Compose it as a subsystem instead.
Subclassing a non-abstract `System` raises `TypeError: cannot subclass ...`. Newer
poincare applies this to every concrete system, including the jablonski transitions
(`class MyFluorescence(Fluorescence)` worked in 1.2.0 and fails now).

## Equations

```python
eq = x.derive() << -k * x       # dx/dt = -k x
force = v.derive() << -(w**2) * x
```

- `derivative << rhs` creates an `Equation`. Assign it to a class attribute (any name).
- **Several equations for the same derivative are summed.** Write each process as its own
  equation (production, degradation, …) rather than one long expression.
- Second-order systems need an explicit `v: Derivative = x.derive(initial=...)`.

## Parameters, constants and functions

- **Parameter defaults may be expressions**: `force: Parameter = assign(default=0.2 * real.sin(omega * t))`.
  Parameters that depend on variables or time are substituted into the equations.
- **Time- or variable-dependent parameters cannot be changed with `with_values`**
  (`ValueError: must recompile...`). Build a new simulator from an instance instead:
  `Simulator(Model(force=real.cos(Model.x)))`.
- Parameters that depend only on other parameters (for example a derived photon flux) *can*
  be changed through the parameters they depend on.
- **Math functions** come from `symbolite`: `from symbolite import real` gives `real.exp`,
  `real.log`, `real.sin`, `real.sqrt`, `real.expm1`, `real.tanh`, and others. Do not use
  `numpy` functions on symbolic components.
- **Pint quantities inside compiled expressions do not compile.** `rate * real.exp(-gap / (1 * u.boltzmann_constant * T))`
  fails with a `SyntaxError` at `Simulator(...)`. Put physical constants in a `Parameter`
  (`kB: Parameter = assign(default=1 * u.boltzmann_constant)`) and use that instead.
  A quantity as a component's *default value* is fine.

## Composition (subsystems)

A `System` can be instantiated inside another one. Keyword arguments link its components
to the outer components, and anything not passed is created per instance.

```python
class Damping(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)
    gamma: Parameter = assign(default=0)
    damping = vx.derive() << -gamma * vx

class DampedOscillator(System):
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)
    k: Parameter = assign(default=1)
    gamma: Parameter = assign(default=0.1)
    spring = vx.derive() << -k * x
    damping = Damping(x=x, gamma=gamma)   # adds its equation to the same derivative
```

- Linked components share one state, and their equations add up.
- Unlinked components are named with the subsystem prefix. A variable shows up as
  `"sub.y"` in the results, and parameters as `sub.k`. Access them as
  `Outer.sub.component`, for example in `with_values({Outer.sub.k: 0.2})`. The output
  contains variables (and transforms), never parameters.
- Passing a number instead of a component sets that instance's default: `Damping(gamma=0.3)`.
- You can reuse one subsystem class many times (chains, networks). Each instance is
  independent unless linked.

## Simulator

```python
import numpy as np
from poincare import Simulator, solvers

sim = (
    Simulator(Model)                    # compiles once; accepts a class or an instance
    .with_solver(solvers.LSODA(rtol=1e-8, atol=1e-10))
    .with_values({Model.k: 3, Model.x: 2})
    .with_times(save_at=np.linspace(0, 10, 101))
    .with_transform({"E": 0.5 * Model.v**2 + 0.5 * Model.x**2}, append=True)
)
result = sim.solve()
```

- **Immutable builder.** Every `with_*` returns a new simulator that shares the compiled
  function. Loops over values are cheap.
- `with_values(mapping, append=True)`: keys are components (`Model.k`, `Model.sub.x`).
  Values are numbers or pint quantities whose units must be compatible with the default's
  (otherwise `DimensionalityError`).
- `with_transform(mapping_or_list, append=False)` controls the output. **Without
  `append=True`, the output contains only the transform entries**, not the variables.
- `with_solver(solver)`: **the only way to choose the solver.** In 1.2.0 the `solver=`
  argument of `Simulator(...)` is accepted but silently ignored. Newer versions remove it
  (`TypeError: unexpected keyword argument 'solver'`).
  - Solvers: `solvers.LSODA` (default), `RK23`, `RK45`, `DOP853`, `Radau`, `BDF`. All are
    frozen dataclasses with `rtol` (default 1e-3), `atol` (1e-6), `first_step`, `max_step`.
  - The defaults give roughly 1e-3 relative error. **Set tolerances explicitly for
    quantitative work**, and for stiff systems (photophysics, kinetics with widely
    different rates) use LSODA, Radau or BDF.
- **`atol` is in SI base units**, because integration runs in base units (mol, m³, s,
  eV → J). Amounts like 1e-16 mol or volumes like 1e-15 m³ (1 pL) are invisible under the
  default `atol=1e-6`. The run then "succeeds" with nonsense (for example product exceeding
  substrate). Set `atol` a few orders below the smallest base-unit magnitude
  (`atol=1e-27` for ~1e-18 mol). Check conservation to confirm.
- `LSODA` with `t_span` starting at 0, a `save_at` and no events uses scipy's `odeint`
  internally. Otherwise it uses `solve_ivp`. **`odeint` can fail silently:** with few,
  widely spaced `save_at` points it hits its 500-step limit per output interval, warns
  `Excess work done` and returns wrong values without raising. Never hide warnings
  (`-W ignore`, `warnings.filterwarnings`) while checking results. If you see that
  warning, NaNs or a garbage trajectory, try `solvers.LSODA(..., implementation="LSODA")` (the
  `solve_ivp` path), `solvers.Radau` or `solvers.BDF`.
- `with_times(t_span=None, save_at=None)`: **the recommended way to set the simulation
  times.** It stores them on the simulator, so they carry through the other `with_*` calls
  and `solve()` needs no arguments. `solve(t_span=..., save_at=...)` still works, and each
  argument not passed to `solve` falls back to its stored value independently.
  `sim.with_times(save_at=t).solve()` equals `sim.solve(save_at=t)`. **Not in 1.2.0:**
  there, pass the times to `solve`.
- **Debugging:** `prob = sim.create_problem()` exposes `prob.y` and `prob.p` (initial state
  and parameters in base units, ordered like `sim.compiled.variables` / `.parameters`)
  and `prob.rhs(t, y, p, dy)`, so you can evaluate the right-hand side by hand.
- `Simulator(Model, backend="numpy" | "numba" | "jax")`. numpy is the default. numba and
  jax must be installed separately (jax is not available on Windows).
- `solve(t_span=None, save_at=None, events=(), check_dimensionality=True)` (times default to
  those set with `with_times`). The rules below hold wherever the times are given:
  - With units, `save_at` must be a time quantity (`np.linspace(0, 10, 101) * u.s`).
    Without units, it must be a plain array.
  - Without `t_span`, the time span is inferred from `save_at`: integration starts at
    `save_at[0]`, and the initial values belong to that time. A `save_at` from 1 to 10
    starts the system at t = 1.
  - To start at 0 but output only later times, include 0 in `save_at` and drop it
    afterwards (works in every version). Passing `t_span=(0, t_end)` also works in newer
    versions. In 1.2.0, the default `LSODA` ignores an explicit `t_span` starting at 0
    when `save_at` starts later.
  - `events` takes `scipy_events` events (`Event(condition=..., terminal=True)`).
- `sim.interact(values={Model.k: (0, 10, 0.1)}, save_at=...)` (newer versions also take the
  times from `with_times`) gives ipywidgets sliders in
  Jupyter. In marimo, put a `mo.ui.slider` value into `with_values` instead.

## Results

`solve` returns an `xarray.Dataset` indexed by `time`, with one data variable per output.
Subsystem variables are named with dots (`"sub.x"`).

```python
result["x"]                                   # DataArray (pint-aware if the model has units)
result.pint.dequantify().to_dataframe()       # strip units → pandas (needed for .plot())
result["x"].pint.to(u.mm).pint.dequantify().values
result.to_dataframe().plot()                  # unitless models
```

The time coordinate keeps the unit of `save_at` (for example ms). In scripts without a
display, save figures (`plt.savefig(...)`) instead of calling `plt.show()`, or set
`MPLBACKEND=Agg`.

## Reaction primitives

`poincare.reactions` defines `Reactant`, `reaction_initial`, `RateLaw`, `MassAction` and
`AbsoluteRateLaw`. simbio re-exports and extends them, and jablonski transitions wrap
`MassAction`. Usage is documented in [simbio.md](simbio.md#reactions).

**Newer than 1.2.0:** a `Parameter` can be a reactant. It acts as a **constant species**
(a buffered reservoir or a fixed substrate): it enters the `MassAction` rate law with its
stoichiometry but gets no equation.

```python
class Model(System):
    G: Parameter = assign(default=2)        # constant species
    x: Variable = initial(default=0)
    k: Parameter = assign(default=3)
    make = MassAction(reactants=[2 * G], products=[x], rate=k)   # dx/dt = k·G²
```

## Reports and introspection

- `print(Model)` or `Model` in a notebook shows a table of variables, parameters and equations.
- `Model.k.default` (parameters and constants) and `Model.x.initial` (variables) give the
  declared defaults, which is useful for analytic checks.
- `Model.variables` and `Model.parameters` return `xarray.DataArray`s of components.
- `poincare.printing.notebook_report(Model, transform=None, descriptions=None,
  replace_algebraics=False, sections=default_notebook_sections, level=2, show=True)`
  renders a Markdown report with equations, a variable table and a parameter table in
  Jupyter or marimo. It takes the same `transform`, `descriptions` and `replace_algebraics`
  as `model_report`. With `show=False` it returns the Markdown string. `level` is the
  heading level of the section titles. Requires poincare newer than 1.2.0 (not on PyPI yet).
  - `sections` maps titles to writers `(model, latex) -> str` that return Markdown, where
    `latex` is the shared `ToLatex`. Extend the defaults with
    `default_notebook_sections | {"Notes": lambda model, latex: "..."}`.
  - simbio's `notebook_report` adds a reactions section and jablonski's
    (`jablonski.plots.notebook_report`) a Jablonski diagram. Use those for their models.
- **Report bug (1.2.0, fixed in newer versions):** reports crash with
  `TypeError: 'NoneType' object cannot be interpreted as an integer` when a variable has
  no equation but appears in other equations. Typically this is a constant `Volume` in a
  simbio `Compartment` with amount species. Give it a zero-rate equation through a
  `Parameter` (`V.derive() << growth_rate`, default `0 * u.pL / u.s`). An inline quantity
  (`V.derive() << 0 * u.pL / u.s`) breaks the report too.
- **Newer than 1.2.0:** the Default column of the variable and parameter tables is real
  LaTeX. Expression defaults are rendered as math, `transform` renames apply inside them,
  and quantities show their units (`3\ \frac{1}{\mathrm{s}}`).
- `model_report(Model, path=None, transform={Model.k: "k_1"}, descriptions=None,
  standalone=True, replace_algebraics=False)` returns (or writes to `path`) a LaTeX report
  with equations, a variable table and a parameter table.
  - simbio's `model_report` adds a reactions section.
  - jablonski's (`jablonski.plots.model_report`) adds a Jablonski diagram.
- Lower-level LaTeX helpers live in `poincare.printing.latex`: `latex_equations`,
  `parameter_table`, `variable_table`, `ToLatex`. On newer poincare they take
  `format="latex" | "markdown"`.
- `poincare.helpers.get_from("sub.k", Model)` resolves a dotted name to a component, and
  `to_values({"sub.k": 2}, Model)` turns a dict of dotted names into a `with_values` mapping.
