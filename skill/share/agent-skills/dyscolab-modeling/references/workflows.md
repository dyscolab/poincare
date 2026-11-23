# Workflows: suggestions, file layout and model conventions

## Workflow suggestions

The stages (declare, check, simulate, analyse, compare, report) combine freely. Common
tasks and what usually makes a good result:

| Task | A good result |
| --- | --- |
| Declare / write / implement a model of X | The model, a model report (`notebook_report` in a notebook) and one short example simulation with a plot |
| Simulate X (under conditions Y) | The model (if missing), that simulation, one plot or summary |
| Find the steady state / sweep / bifurcation / period | That analysis on a declared, checked model |
| Fit to data | A fit to the data the user supplied, with the fitted values and a plot against the data |
| Report / write up the model | `model_report` LaTeX, or a short description |

- **Compare with the user's data** (a file, an array, or numbers in the message). Don't
  invent "typical experimental" curves. If parameter values are missing, use clearly
  labelled placeholders and say so.
- **Sanity checks are part of a correct model.** Keep them cheap: check that units are
  consistent, that conserved totals stay constant, and run one limiting case with a known
  answer. Report the outcome in one line.
- When the work grows, build on the existing files rather than starting over.

## Showing that a declared model works

A model report is the quickest way for the user to check that the code matches the
equations they had in mind. Prefer a notebook (Jupyter or marimo) with:

```python
# cell 1: the report (renders Markdown + LaTeX in Jupyter and marimo)
from poincare.printing import notebook_report  # simbio: from simbio import notebook_report
notebook_report(MyModel)                       # jablonski: from jablonski.plots import notebook_report

# cell 2: one example simulation
result = sim.with_times(save_at=...).solve()
result.pint.dequantify().to_dataframe().plot()
```

- **Run the notebook before handing it over**, so the user gets it with its outputs:
  `jupyter nbconvert --to notebook --execute --inplace analysis.ipynb`, or `nbclient`
  from Python. If the environment has no notebook runner and you can't add one, say so
  and deliver the code as a script that you have run.
- `notebook_report(Model, show=False)` returns the Markdown string instead of displaying
  it, for example to save it to a `.md` file.
- simbio's `notebook_report` starts with a reactions section, and jablonski's
  (`jablonski.plots.notebook_report`) starts with the Jablonski diagram, so use the one from
  the library the model is built with.
- In scripts, `print(model_report(Model))` gives the LaTeX source, or use
  `model_report(Model, path="model.tex")`.
- For a larger model in `model.py`, the notebook imports it (see File layout below), so the
  report always reflects the current model file.

## File layout

**Small models** (a handful of states and equations): put the model and its example run in
one script or notebook.

**Larger models** (many states or parameters, several transitions or reactions, or a model
reused by several analyses): keep the declaration and the analysis apart.

```
model.py            # (or models/…) System classes only: no simulation at import time
analysis.ipynb      # or analysis.py: imports the model, configures Simulators, runs things
```

Making the import work, in this order of preference:

1. If the project already has a package or `pyproject.toml`, put the model in it and
   install the project (editable) into the environment the user already uses.
2. If the model file sits next to the notebook or script, `from model import MyModel` works
   from the notebook directory.
3. If the notebook is in a subfolder, add the model's folder in the first cell:

   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path("..").resolve()))   # folder containing model.py
   from model import MyModel
   ```

Do not create `pyproject.toml`, `src/` layouts, data folders or environment files unless the
user asks for them or they already exist.

**Full research projects** (only when the user is building one): a small package with model
modules and custom transitions or reactions, notebooks per analysis or figure that import it,
and a folder for the *user's* data files.

## Implementing a model from a publication

- **Stay close to the source.** Enter every value in the units and form the source uses
  and let pint convert it: a level given as 545 nm is `initial(545 * u.nm)`, a rate in
  cm³/s stays in cm³/s. Keep relations as the source writes them (a ratio stays
  `5.9 * k_ET31`, a branching ratio stays `0.70 * k_R6`). Don't replace values the
  source gives with your own estimates.
- **Use the source's symbols** for states, parameters and transitions (`n6`, `k_ET13`,
  `k_NR4`), so each line can be checked against the paper. Put the physical name in a
  comment (`n6  # 2H11/2, 4S3/2`).
- **Reference the source.** Cite it in full (authors, title, journal, year, DOI) in the
  module docstring or the first notebook cell. Give each parameter, state and transition
  a comment with where it comes from: `# Table 1`, `# Eq. 3`, `# Fig. 1`, `# p. 39`.
  Values that are not in the source are placeholders and say so (`# PLACEHOLDER, not in
  the paper`).

## Model conventions

- **Write the model directly with the library.** Declare states, parameters and
  transitions with `initial`, `assign` and the library's classes in the class body. Don't
  add aliases for library classes, helper functions that wrap `initial` or `assign`,
  dictionaries of values looked up in the class body, or module-level constants that
  only feed one default. The class should read like a list of the model's parameters and
  its equations (poincare), reactions (simbio) or transitions (jablonski).
- **One place for parameters.** Declare every rate and physical constant as a named
  `Parameter` with units and a short comment giving its meaning and source:
  `k_r6: Parameter = assign(default=1510 / u.s)  # 4S3/2 radiative, Anderson 2013 Table 2`.
  Unknown values get a placeholder and a `# TODO: value not reported` comment.
- **Derived quantities are expressions**, not duplicated numbers:
  `photon_flux = power_density / (h * c / wavelength)`, `initial(default=yb_fraction * (1 - isolated))`.
  Changing the base parameter with `with_values` updates everything that depends on it.
- **Control parameters** (power, temperature, wavelength) are `Parameter`s with defaults,
  so experiments only call `with_values` or set `piecewise` events.
- **Build a subsystem (a small `System` class) when it is used several times or stands
  for a physical or conceptual unit**, and instantiate it with keyword links. For
  example:
  - an `YbIon` class (two levels, absorption, decay) instantiated twice, once for the
    ions that take part in energy transfer and once for isolated ions;
  - an `ErIon` class holding one ion's levels and its internal relaxation and emission,
    with the Yb–Er transfers declared in the model that combines the two ions;
  - a custom transition or reaction motif that appears in many places.
- **Name states and transitions physically** when there is no source notation to follow
  (`Er_4S3_2`, `etu_13`, `green_decay`). When there is one, use it (see
  [Implementing a model from a publication](#implementing-a-model-from-a-publication)).
  The names become the output columns and the per-line keys (`line_<name>` in jablonski 0.4.0,
  `emission_<name>` / `absorption_<name>` in newer versions).

## Running experiments

- Build one base simulator, `Simulator(Model).with_solver(solvers.LSODA(rtol=..., atol=...))`,
  and derive variants with `with_values` / `with_transform(..., append=True)`. Don't
  re-create simulators in loops (that recompiles).
- Time-dependent protocols (pulses, steps, temperature jumps): use jablonski's `piecewise`
  for photophysics. For plain poincare/simbio models, run consecutive `solve` calls,
  passing the final state on with `with_values`, or use a time-dependent `Parameter`
  expression.
- **Simulate long enough for the slowest process.** A time-resolved run should last until
  the slowest decay is over, about 5× the longest lifetime (1/Σk of the slowest state),
  not only until the fastest signals peak.
- Scans: use the loop + `xr.concat` pattern ([analysis.md](analysis.md#parameter-sweeps)),
  and nest it or concat on a second dimension for 2D scans. For jablonski spectra over a
  list of excitations given as pumper fluxes, use the `jablonski.sweeps` helpers instead
  ([jablonski.md](jablonski.md#sweeps)).
- Plot with `result.pint.dequantify().to_dataframe().plot()` or xarray's `.plot()`. Label
  axes with units.

## Validation

- **Analytic limits:** single-exponential decay (τ = 1/Σk), two-level steady state
  `k_abs/(k_abs + k_decay)`, Boltzmann ratios `g·exp(-ΔE/kT)` between fast-exchanging levels,
  quantum yield `k_r/Σk`, Michaelis–Menten initial rate, equilibrium constants.
- **Conservation:** total population of each species or ion, or total mass in closed
  networks. Assert it is constant within the solver tolerance.
- **Scaling laws:** for example an n-photon process scales as Pⁿ at low power.
- **Units:** run the model with units at least once. pint catches inconsistent rate
  dimensions.
- Use `np.testing.assert_allclose` with explicit `rtol`, set solver tolerances well below
  it, and dequantify only after converting to a known unit.
