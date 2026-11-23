# Units (pint) across poincare, simbio and jablonski

Units are optional in poincare and simbio, and required in jablonski: time is in seconds
and rates carry units. When units are used, they are checked and errors are raised, not
warnings.

## One registry

```python
import pint
u = pint.get_application_registry()
```

Always use the application registry. jablonski uses it internally (energies, time,
`pump_from_laser`), and quantities from different registries cannot be combined. Don't
create a `pint.UnitRegistry()`, even though some older tutorials do.

## Rules

- **Dimensionless state variables are fine.** Populations, fractions or counts can be
  plain numbers while rates carry units (`1 / u.day`). Time units are then inferred from
  the rates.
- **All or nothing per model.** If any component has units, the independent variable's
  (time) dimensionality is inferred from the derivatives, which must agree. Mixed
  inconsistent units raise `PintError: Units are defined inconsistently` when the class is
  created.
- **`save_at` must match the model.**
  - A model with units needs a time quantity: `np.linspace(0, 10, 101) * u.s` (any time unit;
    the output time coordinate keeps that unit).
  - A unitless model needs a plain array.
  - A mismatch raises `PintError: save at dimensionality ...`.
- **`with_values` checks compatibility.** A value must have units convertible to the
  default's, otherwise you get `DimensionalityError ... for <component>`. Plain numbers are
  only allowed for dimensionless defaults.
- **Integration runs in SI base units.** Results are converted back to the units of each
  variable's initial value.
- **Unit constants in expressions:** `1 * u.boltzmann_constant` written inline in a rate
  expression does not compile. Put it in a `Parameter` first (see
  [poincare.md](poincare.md#parameters-constants-and-functions)).
- **Results:**
  - `result["x"].pint.to(u.mM)` converts.
  - `result.pint.dequantify()` strips units (do this before `.to_dataframe().plot()` or numpy
    math).
  - `.pint.dequantify().values` gives magnitudes in the stored unit, so convert first if
    you need a specific unit.
- **Rate units:**
  - `MassAction` with `n` reactant molecules needs rate units of
    `1 / (time · concentration^(n-1))`. For example, `A + B -> AB` takes `1 / (mM · s)`.
  - `RateLaw` rate laws are concentration/time.
  - `AbsoluteRateLaw` rate laws are amount/time.
- **simbio compartments:** species declared with `concentration(...)` take concentration
  units, `amount(...)` takes amount units, and `volume(...)` takes volume units. Volume
  conversions are inserted automatically (see [simbio.md](simbio.md#compartments-and-volume)).
- **jablonski:**
  - `SpectroscopicSystem` has `time = Independent(default=0 * u.s)`, so `save_at`, `piecewise`
    event times, pulse widths and `upto_t` must all be time quantities.
  - Rates are `1/s` (`Absorption.rate` is a cross-section in `cm**2`, and `pump` is a photon
    flux in `1/(cm**2 s)`).
  - State energies accept energy, wavelength, wavenumber or frequency (a bare float is eV),
    converted with pint's `"spectroscopy"` context. **A zero energy must be given in eV**
    (`0 * u.eV`), because `0 / u.cm` raises `ZeroDivisionError`.
  - State populations are usually unitless (fractions or relative numbers).
- **rebop stochastic runs:** variable values must be plain integers (copy numbers).
  Rate parameters and `upto_t` may carry units.
- **Removing units (newer than poincare 1.2.0):** `poincare.strip_units(Model)` returns a
  copy of a model class (or instance) with every quantity replaced by its magnitude in SI
  base units, as the simulator uses internally. The removed units are kept in
  `Stripped._units` (`{component: base unit or None}`). Use it to run a model with units
  through code that needs plain numbers. Times, values and results are then in base units
  (s, mol/m³, …), not the declared units.
- **SteadyState** results keep variable units, but the reported `time` is a plain number in
  base units.
