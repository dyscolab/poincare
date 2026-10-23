# AGENTS.md — poincare

Instructions for coding agents working **on** this repository. User
documentation is at https://dyscolab.github.io/poincare. Record user-visible
changes in [CHANGELOG.md](CHANGELOG.md).

## Project overview

poincare lets users declare dynamical systems (ODEs) as Python classes and
simulate them. A `System` subclass declares variables, parameters and
equations as class attributes. The declaration is symbolic: every component
is a symbolite `Real`. `Simulator` compiles a system into a first-order
vectorised right-hand side, using a symbolite numerical backend (numpy,
numba or jax), and integrates it with scipy. Results come back as `xarray`
datasets carrying pint units.

poincare sits between symbolite (below) and simbio and jablonski (above).
`poincare.reactions` already contains the reaction primitives (`RateLaw`,
`MassAction`, `Reactant`) and a stochastic rebop simulator that simbio and
jablonski build on. Typical tasks are changes to declaration semantics,
compilation, the simulator API, units handling, solvers, and the analysis
tools (steady state, oscillations, sweeps, fits).

## Repository map

| Path | Responsibility | Touch when |
| --- | --- | --- |
| `src/poincare/_node.py` | `Node` descriptor (name and parent), `NodeMapper`, `_yield` | Ownership, naming and copying of components between systems |
| `src/poincare/types.py` | `Constant`, `Parameter`, `Variable`, `Derivative`, `Independent`, `Equation`, `EquationGroup`, `System` (metaclass `EagerNamer`), `assign`, `initial` | Declaration semantics, composition, defaults |
| `src/poincare/compile.py` | Equation collection, algebraic substitution, reduction to first order, codegen via symbolite `Block`, `Backend`, `SystemCompiler`, transforms | How systems become `rhs(t, y, p, dy)` |
| `src/poincare/simulator.py` | `Simulator` (immutable builder `with_values` / `with_solver` / `with_transform`), `create_problem`, `solve`, `interact` | Simulation API and result construction |
| `src/poincare/solvers.py` | `Solver` protocol, scipy solvers (`LSODA` default, `RK23`, `RK45`, `DOP853`, `Radau`, `BDF`) | Integrators |
| `src/poincare/units.py` | pint registration, unit inference and checks, `strip_units` / `UnitStripper` / `strip_node` | Units |
| `src/poincare/reactions/` | `Reactant`, `RateLaw`, `MassAction`, `AbsoluteRateLaw`, volume hooks, `rebop/` stochastic simulator | Reaction semantics used by simbio and jablonski |
| `src/poincare/analysis/` | `SteadyState`, `Oscillations`, `Sweeper`, `Fitter`, `fit_result` | Post-simulation analysis |
| `src/poincare/printing/` | LaTeX `model_report`, tables, notebook display | Reports |
| `src/poincare/**/tests/` | Tests (the four `testpaths` in `pyproject.toml`) | Always |
| `examples/` | Example scripts plus `test_examples.py` (**not** in `testpaths`) | Changing public API |

## Architecture and core abstractions

- **Declaration through descriptors.** Components are `Node` *and* symbolite
  `Real`. `__set_name__` gives each one a name and its owning class. Reading
  a component through an instance makes a per-instance copy
  (`_copy_from`), so subsystems can be instantiated several times. Nested
  systems pass components in with keyword arguments.
  `System.__init_subclass__` builds a keyword-only `__signature__` from the
  annotations, and annotations that do not match the attribute type raise
  `TypeError`.

- **Components.** `initial(default=)` creates a `Variable` and
  `assign(default=)` creates a `Parameter`
  (`assign(default=, constant=True)` creates a `Constant`). `x.derive()`
  creates a `Derivative`, and `derivative << rhs` creates an `Equation`.
  `Independent` is the time variable.

- **Compilation** (`compile.py`):

  1. Collect equations. Several equations for the same derivative are
     **summed**, which is how reactions combine.
  1. Parameters whose default depends on variables or time become
     algebraic and are substituted into the equations.
  1. Constants used in equations become parameters.
  1. Higher-order derivatives become first-order state.
  1. The result is emitted as a symbolite `Block` and translated with the
     backend's libsl.

  State and parameter vectors are ordered by `sorted(..., key=str)`.

- **Simulation.** `create_problem` evaluates the defaults and values,
  converts pint quantities to **base-unit magnitudes**, and records an output
  scale for each transform entry. `solve` integrates through the `Solver`
  and rebuilds a pint-aware `xr.Dataset` indexed by `time`.

- **Units.** Unit compatibility is checked when a class is created
  (`infer_independent_units`) and when it is simulated (`save_at`
  dimensionality, value compatibility). `strip_units` returns a unitless
  copy of a system. Each `Node` subtype registers a `units.strip_node`
  handler.

- **Extension points** (`functools.singledispatch`): `symbolite.ops.translate`,
  `types.get_default`, `units.strip_node`, and in `reactions/reactions.py`
  `compensate_volume` and `make_concentration`. simbio registers its own
  `compensate_volume` and `make_concentration` implementations.

## Invariants and constraints

Verified:

- **Keep declaration separate from execution.** Systems stay symbolic. Only
  `Simulator` or `SystemCompiler` choose a backend. Do not evaluate
  expressions numerically inside `types.py`.
- **Equations for the same derivative add up** (`eqsum`). Reaction stoichiometry
  depends on it (`reactions/tests/test_reactions.py`).
- **Units are enforced and raise errors.** Since 1.2.0, a `save_at`
  dimensionality mismatch raises instead of warning. Integration runs in
  base units and results are converted back (`tests/test_units.py`,
  `tests/simulator/test_time.py`). Do not drop pint information or turn
  errors into warnings.
- **Every backend and solver gives the same answer.** `tests/simulator/test_backends.py`
  checks the harmonic oscillator against `cos(t)` for every `Backend` and
  every class in `solvers.__all__`. Subclasses of `solvers._Base` must
  appear in `solvers.__all__`, which is asserted at class creation.
- **`Simulator` is immutable.** `with_*` methods return new simulators and
  share the compiled function. Changing a time-dependent assignment
  requires a new `Simulator` (`create_problem` raises `ValueError`).
- **Components and systems are pickleable** (`tests/test_serialize.py`).
- **A system has exactly one `Independent`**, and more than one raises
  `TypeError`.

Established conventions:

- A new `Node` type registers `translate`, `get_default` (where needed) and
  `units.strip_node`, following `types.py` and `reactions/reactions.py`.
- Each module that imports `pint_xarray` resets
  `pint.get_application_registry().force_ndarray_like = False`. Keep this
  reset, because compilation breaks without it.
- Explicit pint quantities cannot be compiled, so a `RateLaw` or
  `MassAction` with a pint rate wraps it in a generated `Parameter`.

Unresolved (report, do not decide):

- Immutability of components after class creation is not implemented
  (`tests/test_inmutability.py` is xfail).
- Whether instance-level independent-unit checks are needed (TODO in `System.__init__`).

## Development environment and commands

The pixi manifest is in [pyproject.toml](pyproject.toml) under
`[tool.pixi.*]`. These environments install the **published** symbolite
from PyPI or conda-forge. If your change depends on unreleased symbolite
behaviour, run the tests with a source install of symbolite and say so in
your report.

```bash
pixi run test                                  # pytest --doctest-modules (default env: py312, test, rebop)
pixi run -e test-py313 test                    # CI matrix: test-py312, test-py313
pixi run test src/poincare/tests/test_units.py # subset
pixi run -e lint lint                          # pre-commit run --all-files (ruff check --fix, ruff format, mdformat, taplo)
pixi run -e build _build                       # uv build
pytest examples/test_examples.py               # run from this dir inside a test env; not run by `pixi run test`
```

- `pixi run -e lint lint` **rewrites files**, and the tree currently has
  unformatted files and about 10 ruff findings. Run ruff only on the files
  you changed unless asked otherwise
  (`pixi run -e lint ruff check <files>`).
- Pyright, pyrefly and ty are configured in `pyproject.toml`, but no task,
  hook or CI step runs them. Typing is not enforced.
- `conftest.py` writes `test_readme.txt` (gitignored) from the README's
  python blocks. The README currently has none.

Before calling a change done, run `pixi run test`, run ruff on the changed
files, run the examples if the public API changed, and run the simbio and
jablonski test suites (see Ecosystem compatibility).

## Making changes

- **Declaration semantics (`types.py`, `_node.py`):** check composition
  (`test_instantiation.py`, `test_subclass.py`, `test_initials.py`), copying
  through `_copy_from`, and pickling. Read `NodeMapper` before you change
  how parents resolve.
- **Compilation:** keep the ordering, the summing of equations, and the
  handling of algebraic parameters. Run `tests/simulator/` and
  `test_derivative.py` with every backend.
- **New solver:** subclass `_Base` with `solver=<scipy OdeSolver>`, add it to
  `__all__`, and accept `atol`/`rtol`. `test_solvers` then covers it.
- **Simulator API:** keep the builder style. Update `examples/`,
  `analysis/` (which takes `Simulator`s) and the downstream packages.
- **Reactions:** `reactions/reactions.py` changes simbio and jablonski
  behaviour. Keep the `Reactant.from_mul` stoichiometry parsing and the
  volume hooks.
- **Fits (`analysis/fits.py`):** `MAX_RTOL`/`MAX_ATOL` exist because
  `least_squares` takes finite-difference Jacobians. Do not loosen them.

## Testing and scientific correctness

- Prefer analytic references, as in the existing tests: the harmonic
  oscillator against `cos(t)` and exponential decay. Set solver tolerances
  explicitly and well below the assertion tolerance. The convention in
  `test_backends.py` is solver `1e-6` and assertion `1e-3`.
- Check conservation (stoichiometric balances) and limiting cases for
  reaction changes. For units, check that a result is the same with and
  without units (see `test_strip_units.py`).
- When you parametrize tests, cover every `Backend` (numpy, numba, and jax
  on non-Windows).
- Do not loosen tolerances, remove assertions, or add `xfail` to hide a
  regression. Report it instead.

## Ecosystem compatibility

- **Upstream:** poincare depends on `symbolite>=1.1.1`, including internals
  (`core.value`, `core.call`, `impl._lang_value_utils.compile`).
- **Downstream:** simbio uses `System`, `Variable`, `Derivative`, `Node`,
  `NodeMapper`, `Reactant`, `RateLaw`, `compensate_volume`,
  `make_concentration`, `compile.depends_on_at_least_one_variable_or_time`,
  `printing.latex` and `RebopSimulator`. jablonski uses `Simulator`,
  `SteadyState`, `solvers`, `MassAction`, `analysis.fits`
  (`Fitter`, `FitResult`, `UnitsHandler`), `printing.latex`,
  `simulator.Components` and `RebopSimulator`.
- Private-looking modules such as `_node` are effectively public. Before
  changing anything listed above, search the downstream sources for uses.
- **Run the simbio and jablonski test suites for every change**, not only for
  changes to the interfaces listed above. Downstream packages depend on
  compile ordering, equation summing, unit handling and copy semantics, and
  internal changes can break them in unexpected ways.
  - The repos are https://github.com/dyscolab/simbio and
    https://github.com/dyscolab/jablonski.
  - Run their tests in an environment where **this checkout** of poincare is
    installed from source. Their own pixi environments install the published
    poincare.
  - If you cannot run them, say so explicitly in your report.
- Note breaking changes in CHANGELOG.md with a migration path.

## Documentation and public API

- The public API is `poincare.__all__` plus `poincare.reactions`,
  `poincare.analysis`, `poincare.solvers` and `poincare.printing`. The
  package ships `py.typed`, so annotate public signatures.
- Give public classes docstrings that state units and semantics, as
  `RateLaw` and `MassAction` do. Error messages should name the offending
  component and its expected dimensionality.
- Update the examples and CHANGELOG.md when you change the API.

## Completion checklist

- [ ] Understood which stage (declaration, compile, simulate, analyse) the change belongs to.
- [ ] Equation summing, deterministic ordering, unit checks and immutability of `Simulator` are preserved.
- [ ] New `Node` types register `translate`, `strip_node` and `get_default`.
- [ ] Tests compare against analytic or independent references, with explicit tolerances.
- [ ] `pixi run test` passes, ruff is clean on the changed files, and the examples run if the API changed.
- [ ] simbio and jablonski tests pass against this checkout (for every change), or the user is told they were not run.
- [ ] CHANGELOG.md updated, and assumptions or limitations reported to the user.
