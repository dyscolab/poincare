---
name: dyscolab-modeling
description: Declare, simulate and analyse dynamical models in Python with the Dyscolab libraries - poincare (ODE systems, steady states, sweeps, oscillations, fits), simbio (chemical reaction networks, compartments and volumes, SBML/BioModels import, stochastic Gillespie runs) and jablonski (photophysics - Jablonski diagrams, absorption/fluorescence/phosphorescence, energy transfer and upconversion, spectra, quantum yields, pulsed excitation). Use when the user wants to model kinetics, reaction networks, ODEs or photophysical systems, or mentions poincare, simbio or jablonski.
license: MIT
compatibility: Python >= 3.12. Documents the poincare, simbio and jablonski versions it was installed from (see Versions and environment).
---

# Dyscolab modeling

poincare declares ODE systems as Python classes and simulates them. simbio (reaction
networks) and jablonski (photophysics) build on it, so their models are poincare `System`s
and share the same simulator, units and analysis tools.

<!-- dyscolab-skill:versions-section:start -->
## Versions and environment

This skill is **bundled with the libraries**. poincare ships `SKILL.md` and the poincare
references and examples. simbio and jablonski add their own (`simbio*`, `jablonski*` in
`references/` and `examples/`) when they are installed, so links to a library that is not
installed lead nowhere. The skill documents these versions:

<!-- dyscolab-skill:installed-versions:start -->
The versions installed in the Python environment this skill folder belongs to
(`<environment>/share/agent-skills/dyscolab-modeling`, which `pixi skills` links into the
agent's skills folder).
<!-- dyscolab-skill:installed-versions:end -->

**Check it against the environment you work in** before writing code, with that
environment's Python:

```sh
python -m poincare.skill check <path of this skill folder>
```

It prints the installed library versions and whether this folder matches them. If it does
not (a library was upgraded, added or removed):

- **Copied with `python -m poincare.skill install`** (or `simbio.skill`/`jablonski.skill`):
  run `python -m poincare.skill install --dest <folder that contains this skill folder>`
  again. It replaces this folder.
- **Linked by `pixi skills`:** the link follows the environment, so it is always current
  for that environment. If you work in a different environment, run `pixi skills manage`
  there or use `install` as above.
- **No `poincare.skill` module:** the installed poincare is older than the bundled skill.
  Read the installed sources, or use the separate skill from
  [github.com/dyscolab/skills](https://github.com/dyscolab/skills) that matches the
  installed releases.

**Installing the libraries:** poincare and jablonski are published on PyPI and
conda-forge, simbio and rebop (stochastic runs) on PyPI. simbio's SBML import needs its
`io` extra (`simbio[io]`). Install them with the tool that manages the project's
environment (pixi, conda/mamba, uv, pip, …), and don't mix tools in one environment.

Notes marked "newer than" a release describe what changed after that release. If the
installed source disagrees with this skill, the installed source wins.
<!-- dyscolab-skill:versions-section:end -->

## Before writing code

1. **Pick the library.**

   | The system is… | Use | Reference |
   | --- | --- | --- |
   | Any ODE: mechanics, population dynamics, circuits, generic kinetics written as equations | poincare | [references/poincare.md](references/poincare.md) |
   | Chemical/biochemical reactions, mass action, enzymes, compartments with volumes, SBML/BioModels, stochastic copy numbers | simbio | [references/simbio.md](references/simbio.md) |
   | Electronic states and photophysical transitions: absorption, emission, ISC, energy transfer, upconversion, spectra, lifetimes, quantum yields | jablonski | [references/jablonski.md](references/jablonski.md) |

   They mix freely: a jablonski model can contain plain poincare equations, and the analysis
   tools in [references/analysis.md](references/analysis.md) work for all three.
2. **Use units** (pint) when the user gives physical values. jablonski always needs them.
   Read [references/units.md](references/units.md) first.

## Core pattern

```python
import numpy as np
import pint
from poincare import System, Variable, Parameter, Simulator, initial, assign, solvers

u = pint.get_application_registry()   # always this registry, never pint.UnitRegistry()

class Decay(System):
    x: Variable = initial(default=1 * u.mM)     # annotate every component
    k: Parameter = assign(default=0.5 / u.s)
    decay = x.derive() << -k * x                 # dx/dt = -k x

sim = (
    Simulator(Decay)
    .with_solver(solvers.LSODA(rtol=1e-8, atol=1e-12))
    .with_times(save_at=np.linspace(0, 10, 101) * u.s)
)
result = sim.solve()   # xarray Dataset with units
result.pint.dequantify().to_dataframe().plot()
```

Declare (a symbolic class) → `Simulator` (compiles once) → `with_values` / `with_transform` /
`with_solver` / `with_times` (cheap, immutable copies) → `solve()` → xarray `Dataset`.
Set the times with `with_times(save_at=...)` (`RebopSimulator`:
`with_times(upto_t=..., n_points=...)`). poincare 1.2.0 has no `with_times`: there, pass the
same arguments to `solve` (`sim.solve(save_at=...)`), which newer versions still accept.

## Rules that prevent most errors

- **Equations for the same derivative add up.** Write one equation per process.
- **`Simulator` is immutable.** `with_*` returns a new one. Set the solver only with
  `.with_solver(...)`: in 1.2.0, `Simulator(..., solver=...)` is silently ignored, and
  newer versions reject it.
- **Integration starts at `save_at[0]`** (the time span is inferred from `save_at` unless
  `t_span` is given), so
  the initial values belong to that time. Start `save_at` at 0 when they belong to t = 0.
- **The default tolerances (rtol 1e-3) are too loose for quantitative work**, and steady
  states inherit them. Use `solvers.LSODA(rtol=1e-8, atol=...)` with `atol` far below the
  smallest value you care about, **measured in SI base units** (1 pL = 1e-15 m³ and
  1e-18 mol are tiny numbers). Check a conserved total to catch a bad tolerance.
- **`LSODA` can fail silently.** From t = 0 with `save_at` and no events it uses scipy's
  `odeint`. With few, widely spaced `save_at` points that hits a step limit, warns
  `Excess work done` and returns wrong values. Use `solvers.LSODA(..., implementation="LSODA")`
  for long runs, and never hide warnings (`-W ignore`) while checking results.
- **Units are all or nothing.** With units, `save_at`, event times and `upto_t` are time
  quantities, and `with_values` must use compatible units. Without units, use plain
  numbers. Put physical constants in a `Parameter`, because inline pint constants in rate
  expressions don't compile. Call `.pint.dequantify()` before plotting.
- **Use `symbolite.real` functions** (`real.exp`, `real.sin`, …), not numpy, inside model
  expressions.
- **`with_transform(...)` replaces the output** unless you pass `append=True`.
- **Parameters that depend on time or on variables cannot be changed with `with_values`.**
  Build a new `Simulator(Model(param=new_expr))` instead.
- **simbio:**
  - `SteadyState`, `Oscillations` and `solvers` come from `poincare`.
  - A `Compartment` has exactly one `Volume`.
  - With *amount* species, write `MassAction`/`RateLaw` directly in the compartment.
    `simbio.reactions` motifs skip volume conversion.
  - SBML import in simbio 1.2.0 needs Python ≥ 3.14.
- **jablonski:**
  - Helpers take a `Simulator`, not a model class.
  - Transition states are passed by keyword, from source to target.
  - `Absorption.rate` is a cross-section (cm²) and `pump` a photon flux (1/(cm² s)).
  - A zero state energy must be `0 * u.eV`.
  - **Never put a `piecewise` event at exactly t = 0** (0.4.0 applies it late). Start at a
    small positive time, or set always-on excitation with `with_values`.
  - Simulation helpers are named `*_emission` in 0.4.0. Newer versions add generic names
    (`spectral_time_resolved`, `spectra`, `quantum_yield`, …) and keep the `*_emission`
    ones.
  - **Renamed after 0.4.0:** per-line outputs `line_<transition>` became
    `emission_<transition>`, and `EnergyTransferUpconversion4` became
    `EnergyTransferUpconversion`. See [references/jablonski.md](references/jablonski.md)
    for code that runs on both.
- **Sweeps:** for any parameter, use the loop `xr.concat([SteadyState().solve(sim.with_values({M.p: v})).isel(time=-1) for v in vals], dim="p")`.
  The built-in `SteadyState.sweep` fails with units on variables or with transforms.
  For jablonski spectra over a list of pumper excitations, use the `jablonski.sweeps`
  helpers.

## Trust the libraries, check the declaration

- **Assume the libraries are correct.** A correctly declared model generates the right
  equations and simulates correctly. When results look wrong, look for the cause in the
  model first: a wrong sign or rate, a missing or duplicated process (equations for the
  same derivative add up), units, initial values, tolerances or the time span. Don't
  re-derive the equations by hand, write a parallel scipy/numpy implementation "to
  check", or patch around the library.
- **Compare equations against a reference with the generated report**, not by reading the
  code. When checking a model against a paper, textbook or the user's equations, render
  `notebook_report(Model)` in a notebook (`show=False` returns the Markdown string) or
  `print(model_report(Model))` in a script, and compare that output term by term with the
  reference. Use the library's report (simbio and jablonski add reactions and the
  Jablonski diagram). The report shows what the library actually integrates.
- **If something still looks like a library bug** (the report shows the intended
  equations but the simulation contradicts them, a crash inside the library, results
  that change with an unrelated setting), say so, keep the smallest reproducing example,
  and propose to the user to open a bug report at `github.com/dyscolab/<library>/issues`
  with that example, the installed versions and the expected and actual results. Use a
  documented workaround meanwhile, if there is one.

## Workflow suggestions

The stages below are building blocks, not a fixed pipeline: combine the ones the task
needs. Some patterns that work well with poincare and its downstream libraries:

- **Declaring a model** → show that the code matches the intended equations. A notebook
  works well:
  1. One cell with a **model report**: `notebook_report(Model)` from `poincare.printing`
     renders the equations, variables and parameters. For simbio models use
     `simbio.notebook_report` (adds the reactions), for jablonski models
     `jablonski.plots.notebook_report` (adds the Jablonski diagram).
  2. One cell with a short example simulation and a plot.

  Run the notebook before handing it over (`jupyter nbconvert --execute --inplace` or
  `nbclient`). If you can't, say so and give a script you have run.
- **Implementing a model from a paper** → stay close to the source: its units (a level
  given in nm is `initial(545 * u.nm)`), its symbols and its relations. Cite it, and
  give each value a comment with its table or equation. Details:
  [workflows.md](references/workflows.md#implementing-a-model-from-a-publication).
- **Writing any model** → declare states, parameters and transitions directly with
  `initial`, `assign` and the library's classes, without aliases or wrapper helpers. Make
  a subsystem when it is used several times or is a physical unit (an ion, a molecule).
  Details: [workflows.md](references/workflows.md#model-conventions).
- **Time-resolved runs** last until the slowest process is over (about 5× the longest
  lifetime).
- **Simulating a scenario** → the simulation plus one plot or summary of the quantity of
  interest.
- **Steady states, sweeps, oscillations, fits** → build them on top of a declared and
  checked model ([analysis.md](references/analysis.md)).
- **Comparing with data** → use data the user provides. Don't invent "typical"
  experimental curves. When parameter values are missing, use clearly labelled
  placeholders and say so.
- In poincare 1.2.0, reports of a `Compartment` with a constant `Volume` crash
  (`model_report`; fixed in newer versions). Give the volume a zero-rate equation:
  `growth_rate: Parameter = assign(default=0 * u.pL / u.s)` and
  `growth = V.derive() << growth_rate`.
- Cheap sanity checks fit every model: consistent units, conserved totals, one
  limiting case. Report them in a line.

Typical stages:

1. Declare.
2. Sanity-check.
3. Simulate.
4. Analyse: steady states, sweeps, bistability, oscillations, fits, stochastic runs.
5. Compare with the user's data.
6. Report: `notebook_report` in notebooks, `model_report` for LaTeX documents.

## Files

- **Small models:** the model and its example run in one script or notebook.
- **Larger models:** declaration in its own module (`model.py` or `models/`, no
  simulation at import), simulations and analysis in a separate script or notebook that
  imports it.
  - Use the existing package if there is one.
  - Otherwise keep `model.py` next to the notebook, or add its folder to `sys.path` in the
    first cell.
  - Don't create `pyproject.toml`, `src/` layouts or data folders unless asked.

Details and model conventions (parameters with units and sources, derived expressions,
reusable subsystems): [references/workflows.md](references/workflows.md).

## References and examples

- [references/poincare.md](references/poincare.md): components, equations, composition,
  Simulator, solvers, results, reports.
- [references/analysis.md](references/analysis.md): steady states, sweeps, bistability,
  oscillations, fits, rebop.
- [references/units.md](references/units.md): pint rules for all three libraries.
- [references/simbio.md](references/simbio.md): reactions, motifs, compartments, SBML,
  BioModels, graphs.
- [references/jablonski.md](references/jablonski.md): states, transitions, custom
  transitions, excitation, spectra, quantum yields, fits.
- [references/workflows.md](references/workflows.md): workflow suggestions, file layout, conventions,
  validation.
- [references/api-inventory.md](references/api-inventory.md),
  [references/simbio-api.md](references/simbio-api.md),
  [references/jablonski-api.md](references/jablonski-api.md): every public name, internal
  helpers, and what does *not* exist in older versions.
- `examples/`: runnable, tested scripts that each end with physics checks:
  - `poincare_oscillator.py`, `poincare_steady_state_sweep.py`
  - `simbio_compartments.py`, `simbio_sbml.py`, `simbio_stochastic.py`
  - `jablonski_quantum_yield.py`, `jablonski_upconversion.py`

  Copy their patterns.
