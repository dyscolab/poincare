import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import pint
import xarray as xr
from scipy.optimize import least_squares

from .._node import Node
from ..simulator import Components, Simulator
from ..types import Initial, Number

Output = xr.DataArray | xr.Dataset
P0 = Mapping[Components, Initial | tuple[Initial | None, Initial, Initial] | None]

# least_squares estimates the Jacobian by finite differences with a relative
# step of ~1.5e-8, so the solver error must be below that for a reliable fit.
MAX_RTOL = 1e-8
MAX_ATOL = 1e-10
SUGGESTED_RTOL = 1e-10
SUGGESTED_ATOL = 1e-12
# Parameters correlated beyond this are reported as not identifiable.
MAX_CORRELATION = 1 - 1e-6


class ToleranceWarning(UserWarning):
    """The solver tolerances are too loose for a reliable fit."""


class CovarianceWarning(UserWarning):
    """The covariance of the fitted parameters is unreliable."""


class Covariance:
    """Covariance matrix of the fitted parameters.

    Index it by a pair of parameters or their names, pcov["x", "y"], to get
    the covariance in units of x * y. np.asarray(pcov) gives the plain matrix,
    ordered as pcov.parameters, in the units of the fitted values.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        parameters: Sequence[Components],
        units: Sequence[pint.Unit | None],
    ):
        self.matrix = matrix
        self.parameters = list(parameters)
        self.units = list(units)
        self._index = {str(param): i for i, param in enumerate(self.parameters)}

    def _get_index(self, param: Components | str) -> int:
        try:
            return self._index[str(param)]
        except KeyError:
            raise KeyError(f"{param} is not a fitted parameter") from None

    def __getitem__(self, key: tuple[Components | str, Components | str]):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError('index with a pair of parameters, e.g. pcov["x", "y"]')
        i, j = (self._get_index(param) for param in key)
        value = self.matrix[i, j]
        for unit in (self.units[i], self.units[j]):
            if unit is not None:
                value = value * unit
        return value

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.asarray(self.matrix, dtype=dtype)

    def __repr__(self) -> str:
        names = [str(param) for param in self.parameters]
        return f"Covariance(parameters={names},\n{self.matrix})"


class FitResult(dict):
    """Fitted parameter values, as a dict {parameter: value}, and their
    covariance matrix in .pcov (see Covariance).
    """

    # TODO: maybe add? perr (standard errors with units), correlation matrix,
    # cost and cost per target, residuals per target, the raw OptimizeResult.

    def __init__(self, values: Mapping[Components, Initial], *, pcov: Covariance):
        super().__init__(values)
        self.pcov = pcov


def get_default_or_inital(obj: Node) -> Initial:
    try:
        return obj.default
    except AttributeError:
        return obj.initial


class UnitsHandler:
    """Converts between quantities and the plain arrays used by the optimizer.

    Parameter units are taken from p0 (or the model defaults). Variable units
    are taken from each target, which can be a Dataset or a DataArray.
    """

    def __init__(
        self,
        fit_parameters: Sequence[Components],
        clean_p0: Mapping[Components, Initial],
        targets: Sequence[Output],
    ) -> None:
        self.param_units = {}
        for param in fit_parameters:
            default = clean_p0.get(param, get_default_or_inital(param))
            if isinstance(default, pint.Quantity):
                self.param_units[param] = default.units
            else:
                self.param_units[param] = None

        # For a Dataset target: {variable: units}. For a DataArray target: units.
        self.target_units = [self.get_units(target) for target in targets]
        self.target_names = [getattr(target, "name", None) for target in targets]

    @staticmethod
    def get_units(obj: Output) -> pint.Unit | None | dict[Components | str, pint.Unit | None]:
        if isinstance(obj, xr.Dataset):
            return {var: obj[var].pint.units for var in obj.data_vars}
        elif isinstance(obj, xr.DataArray):
            return obj.pint.units
        else:
            raise TypeError(
                f"targets must be xarray Datasets or DataArrays, got {type(obj)}"
            )

    def quantify_parameter(
        self, obj: Initial | Sequence[Initial], param: Components
    ) -> Initial | Sequence[Initial]:
        unit = self.param_units[param]
        if unit is not None:
            return obj * unit
        else:
            return obj

    def dequantify_parameter(
        self, obj: Initial | Sequence[Initial], param: Components
    ) -> Initial | Sequence[Initial]:
        if isinstance(obj, pint.Quantity):
            try:
                return obj.to(self.param_units[param]).magnitude
            except pint.DimensionalityError as err:
                raise pint.PintError(f"Unexpected units in {param}: {err}")
            except AttributeError:
                raise pint.PintError(f"units given in {param} when none were excpected")
        else:
            return obj

    def dequantify_bounds(
        self,
        bounds: Mapping[Components, tuple[Initial, Initial]],
        param: Components,
        i: int,
    ) -> tuple[Number]:
        try:
            return self.dequantify_parameter(bounds[param][i], param)
        except KeyError:
            return (-1) ** (i + 1) * np.inf

    def dequantify_output(self, obj: Output, i: int) -> np.ndarray:
        """Select the variables of target i from obj and return them as a plain
        array in the units of target i.

        For a Dataset target the result has shape (n_variables, ...).
        """
        units = self.target_units[i]
        if isinstance(units, dict):
            if not isinstance(obj, xr.Dataset):
                raise TypeError(
                    f"target {i} is a Dataset, but the simulation function returned {type(obj)}"
                )
            return np.stack(
                [
                    self.dequantify_array(self.select(obj, var), unit, var)
                    for var, unit in units.items()
                ]
            )

        name = self.target_names[i]
        if isinstance(obj, xr.Dataset):
            if name is None:
                raise ValueError(
                    f"target {i} is an unnamed DataArray, the simulation function must return a DataArray"
                )
            obj = self.select(obj, name)
        return self.dequantify_array(obj, units, name)

    @staticmethod
    def select(ds: xr.Dataset, var: Components | str) -> xr.DataArray:
        try:
            return ds[var]
        except KeyError:
            return ds[str(var)]

    @staticmethod
    def dequantify_array(
        obj: xr.DataArray, unit: pint.Unit | None, var: Components | str | None
    ) -> np.ndarray:
        obj_unit = obj.pint.units
        if obj_unit is None and unit is None:
            return obj.to_numpy()
        # A missing unit is treated as dimensionless
        ureg = pint.get_application_registry()
        try:
            return (
                ureg.Quantity(obj.pint.dequantify().to_numpy(), obj_unit or "")
                .to(unit or "")
                .magnitude
            )
        except pint.DimensionalityError as err:
            raise pint.PintError(f"Unexpected units in {var}: {err}")

    def dequantify_sigma(
        self, obj: Output | None, i: int, shape: tuple[int, ...]
    ) -> np.ndarray | None:
        """Return the experimental errors of target i as a plain array in its
        units, with the shape of the dequantified target, or None if not given.

        obj must have the same form as target i. Variables missing from a
        Dataset are padded with ones. Units must have the same dimensionality
        as those of the target.
        """
        if obj is None:
            return None

        units = self.target_units[i]
        if not isinstance(units, dict):
            if not isinstance(obj, xr.DataArray):
                raise TypeError(
                    f"target {i} is a DataArray, so its sigma must be a DataArray, got {type(obj)}"
                )
            name = self.target_names[i]
            return self._broadcast_sigma(
                self.dequantify_array(obj, units, name), shape, name
            )

        if not isinstance(obj, xr.Dataset):
            raise TypeError(
                f"target {i} is a Dataset, so its sigma must be a Dataset, got {type(obj)}"
            )
        given = {str(var): var for var in obj.data_vars}
        extra = given.keys() - {str(var) for var in units}
        if extra:
            raise ValueError(
                f"sigma of target {i} has variables not in the target: {sorted(extra)}"
            )
        if not given:
            return None
        return np.stack(
            [
                self._broadcast_sigma(
                    self.dequantify_array(obj[given[str(var)]], unit, var),
                    shape[1:],
                    var,
                )
                if str(var) in given
                else np.ones(shape[1:])
                for var, unit in units.items()
            ]
        )

    @staticmethod
    def _broadcast_sigma(
        sigma: np.ndarray, shape: tuple[int, ...], var: Components | str | None
    ) -> np.ndarray:
        try:
            return np.broadcast_to(sigma, shape)
        except ValueError:
            raise ValueError(
                f"sigma of {var} has shape {sigma.shape}, but the target has shape {shape}"
            ) from None

    @staticmethod
    def get_save_at(obj: Output) -> Sequence | pint.Quantity:
        index = obj[next(iter(obj.coords))]
        if index.pint.units is not None:
            return index.pint.dequantify().to_numpy() * index.pint.units
        else:
            return index.to_numpy()


class Fitter:
    """Fit parameters of a Simulator to one or more targets simultaneously.

    Parameters
    ----------
    sim
        Simulator whose values are replaced by the parameters being fitted.
    simulation_functions
        One function per target. Each takes a Simulator (with the current
        parameter values) and returns a Dataset or DataArray comparable to its
        target, e.g. ``lambda sim: sim.solve(save_at=t)``.
    targets
        Data to fit. A Dataset target is compared variable by variable to the
        Dataset returned by its simulation function. A DataArray target is
        compared to the returned DataArray, or to the variable with the same
        name if a Dataset is returned.
    p0
        Parameters to fit, mapped to an initial guess, ``None`` (use the
        model default) or a tuple ``(initial guess | None, lower, upper)``.
        If empty, all model parameters are fitted.
    scales
        Factor multiplying the residuals, i.e. how much each variable is valued
        in the fit. A number, or a mapping from variable to number (missing
        # variables default to 1), applied to all targets; or a sequence of
        those, one per target.
    # sigma
        Experimental error (standard deviation) of the targets: a list with,
        for each target, a Dataset or DataArray of the same form (a Dataset may
        leave out variables without errors, which get sigma = 1 in the units of
        the target), or None for no errors. Its units must have the same
        dimensionality as those of the target. The residuals are divided by sigma and
        the covariance treats it as absolute. If None, the noise is assumed to
        be the same for all points and is estimated from the residuals.

    The fit minimizes sum(((y - target) * scale / sigma) ** 2). With scales other
    than 1, the covariance accounts for the scales not being the inverse
    errors (sandwich estimator); with all scales equal to 1 it is the same as
    scipy.optimize.curve_fit.
    """

    def __init__(
        self,
        sim: Simulator,
        simulation_functions: Iterable[Callable[[Simulator], Output]],
        targets: Iterable[Output],
        p0: P0 = {},  # read only
        scales: Number | Mapping[Components | str, Number] | Sequence[Number | Mapping[Components | str, Number]] = 1,
        sigma: Sequence[Output | None] | None = None,
        *,
        units_handler: type[UnitsHandler] = UnitsHandler,
    ):
        self.sim = sim
        self.simulation_functions = tuple(simulation_functions)
        self.targets = tuple(targets)
        if len(self.simulation_functions) != len(self.targets):
            raise ValueError(
                f"got {len(self.simulation_functions)} simulation functions for {len(self.targets)} targets"
            )

        self.p0 = MappingProxyType(dict(p0))
        if p0:
            self.fit_parameters = list(p0.keys())
        else:
            self.fit_parameters = list(sim.model.parameters.values)
        self._clean_p0, self._bounds_p0 = parse_p0(p0)

        self._units = units_handler(self.fit_parameters, self._clean_p0, self.targets)
        self._y0 = [
            self._units.dequantify_output(target, i)
            for i, target in enumerate(self.targets)
        ]
        self._scales = self._parse_scales(scales)
        self._sigma = self._parse_sigma(sigma)
        if self._sigma is None:
            self._weights = self._scales
        else:
            self._weights = [
                scale if sigma is None else scale / sigma
                for scale, sigma in zip(self._scales, self._sigma)
            ]
        self._x0 = self._make_x0(self._clean_p0)
        self._bounds = self._make_bounds(self._bounds_p0)

    def _parse_scales(self, scales) -> list[Number | np.ndarray]:
        if isinstance(scales, Mapping) or np.isscalar(scales):
            scales = [scales] * len(self.targets)
        else:
            scales = list(scales)
            if len(scales) != len(self.targets):
                raise ValueError(
                    f"got {len(scales)} scales for {len(self.targets)} targets"
                )

        parsed = []
        for i, scale in enumerate(scales):
            if not isinstance(scale, Mapping):
                parsed.append(scale)
                continue
            scale = {str(k): v for k, v in scale.items()}
            units = self._units.target_units[i]
            if isinstance(units, dict):
                # Broadcast over the remaining dimensions of the variables
                shape = (len(units),) + (1,) * (self._y0[i].ndim - 1)
                parsed.append(
                    np.reshape([scale.get(str(var), 1) for var in units], shape)
                )
            else:
                parsed.append(scale.get(str(self._units.target_names[i]), 1))
        return parsed

    def _parse_sigma(self, sigma) -> list[np.ndarray | None] | None:
        if sigma is None:
            return None
        if not isinstance(sigma, list | tuple):
            raise TypeError(
                "sigma must be a list with one Dataset, DataArray or None per target"
            )
        if len(sigma) != len(self.targets):
            raise ValueError(
                f"got {len(sigma)} sigma for {len(self.targets)} targets"
            )

        parsed = [
            self._units.dequantify_sigma(s, i, self._y0[i].shape)
            for i, s in enumerate(sigma)
        ]
        if all(s is None for s in parsed):
            return None
        if any(np.any(s <= 0) for s in parsed if s is not None):
            raise ValueError("sigma must be positive")
        return parsed

    def _make_x0(self, clean_p0: Mapping[Components, Initial]) -> np.ndarray:
        return np.array(
            [
                self._units.dequantify_parameter(
                    clean_p0.get(param, get_default_or_inital(param)), param
                )
                for param in self.fit_parameters
            ]
        )

    def _make_bounds(
        self, bounds: Mapping[Components, tuple[Initial, Initial]]
    ) -> tuple[np.ndarray, np.ndarray]:
        return tuple(
            np.array(
                [
                    self._units.dequantify_bounds(bounds=bounds, param=param, i=i)
                    for param in self.fit_parameters
                ]
            )
            for i in range(2)
        )

    def f(self, x: Sequence[Number]) -> np.ndarray:
        """Residuals (y - target) * scale / sigma of all targets for the
        dequantified parameters x."""
        sim = self.sim.with_values(
            {
                param: self._units.quantify_parameter(x[i], param)
                for i, param in enumerate(self.fit_parameters)
            }
        )
        residuals = []
        for i, func in enumerate(self.simulation_functions):
            y = self._units.dequantify_output(func(sim), i)
            residuals.append(((y - self._y0[i]) * self._weights[i]).ravel())
        return np.concatenate(residuals)

    def _check_tolerances(self) -> None:
        solver = self.sim.solver
        rtol = getattr(solver, "rtol", None)
        atol = getattr(solver, "atol", None)
        if rtol is None or atol is None:
            return
        rtol, atol = np.max(rtol), np.max(atol)
        if rtol <= MAX_RTOL and atol <= MAX_ATOL:
            return

        suggestion = (
            f"{type(solver).__name__}("
            f"rtol={min(rtol, SUGGESTED_RTOL):g}, atol={min(atol, SUGGESTED_ATOL):g})"
        )
        warnings.warn(
            f"The current tolerances (rtol={rtol:g}, atol={atol:g}) are too loose "
            f"for fitting (need rtol <= {MAX_RTOL:g} and atol <= {MAX_ATOL:g}); "
            "tighten the solver tolerances with sim.with_solver(poincare.solvers."
            f"{suggestion}) before creating the Fitter, or pass a larger diff_step "
            "(e.g. diff_step=1e-4) to fit. "
            "scipy.least_squares estimates the Jacobian with finite differences, and "
            "solver errors larger than its step can stop the fit early at wrong values.",
            ToleranceWarning,
            stacklevel=3,
        )

    def _jacobian(self, x: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Central-difference Jacobian of f at x, for the covariance.

        The Jacobian from least_squares uses a relative step of ~1.5e-8, which
        leaves solver errors of order rtol / 1.5e-8 in it. A step of rtol^(1/3)
        balances those errors against the truncation error of the central
        difference. Steps are clipped to the bounds (one-sided at a bound).
        """
        rtol = getattr(self.sim.solver, "rtol", None)
        if rtol is None:
            rtol = np.finfo(float).eps
        rel_step = np.cbrt(np.max(rtol))
        lower, upper = bounds

        columns = []
        for j, xj in enumerate(x):
            h = rel_step * (abs(xj) if xj != 0 else 1)
            x_minus, x_plus = x.copy(), x.copy()
            x_minus[j] = max(xj - h, lower[j])
            x_plus[j] = min(xj + h, upper[j])
            columns.append(
                (self.f(x_plus) - self.f(x_minus)) / (x_plus[j] - x_minus[j])
            )
        return np.stack(columns, axis=1)

    def _covariance(self, solution, bounds) -> tuple[np.ndarray, np.ndarray]:
        """Covariance matrix and a mask of the parameters not determined by the
        data (in the null space of a rank-deficient Jacobian)."""
        jac = self._jacobian(solution.x, bounds)
        n, p = jac.shape

        # Pseudo-inverse of A = J^T J, as in scipy.optimize.curve_fit
        _, s, VT = np.linalg.svd(jac, full_matrices=False)
        keep = s > np.finfo(float).eps * max(jac.shape) * s[0]
        A_pinv = (VT[keep].T / s[keep] ** 2) @ VT[keep]

        if all(np.all(np.asarray(scale) == 1) for scale in self._scales):
            scale = None
            pcov = A_pinv
        else:
            # Sandwich estimator: the scales are not the inverse errors
            scale = np.concatenate(
                [
                    np.broadcast_to(scale, y0.shape).ravel()
                    for scale, y0 in zip(self._scales, self._y0)
                ]
            )
            scaled_jac = jac * scale[:, None]
            pcov = A_pinv @ (scaled_jac.T @ scaled_jac) @ A_pinv

        if self._sigma is None:
            # Estimate a common noise from the unscaled residuals
            residuals = solution.fun
            if scale is not None:
                residuals = residuals[scale != 0] / scale[scale != 0]
            dof = residuals.size - p
            if dof > 0:
                pcov = pcov * (residuals @ residuals) / dof
            else:
                warnings.warn(
                    "Not enough data points to estimate the noise without sigma.",
                    CovarianceWarning,
                    stacklevel=3,
                )
                pcov = np.full_like(pcov, np.inf)
        # Remove rounding asymmetry from the matrix products
        pcov = (pcov + pcov.T) / 2

        # The pseudo-inverse gives zero variance along directions the data does
        # not determine; mark the parameters involved in them as unbounded.
        undetermined = np.any(np.abs(VT[~keep]) > 1e-6, axis=0)
        pcov[undetermined, :] = np.inf
        pcov[:, undetermined] = np.inf
        return pcov, undetermined

    def _make_result(self, solution, bounds) -> FitResult:
        at_bounds = [
            str(param)
            for param, active in zip(self.fit_parameters, solution.active_mask)
            if active
        ]
        if at_bounds:
            warnings.warn(
                f"{at_bounds} ended at a bound, their covariance is unreliable.",
                CovarianceWarning,
                stacklevel=3,
            )

        pcov, undetermined = self._covariance(solution, bounds)
        std = np.sqrt(np.diag(pcov))
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = pcov / np.outer(std, std)

        # With a finite-difference Jacobian, degenerate parameters rarely make it
        # exactly rank deficient, but show up as an almost perfect correlation.
        correlated = [
            (str(self.fit_parameters[i]), str(self.fit_parameters[j]))
            for i, j in zip(*np.triu_indices(len(std), k=1))
            if abs(correlation[i, j]) > MAX_CORRELATION
        ]
        undetermined = [
            str(param) for param, u in zip(self.fit_parameters, undetermined) if u
        ]
        if undetermined or correlated:
            details = []
            if undetermined:
                details.append(f"undetermined, covariance set to inf: {undetermined}")
            if correlated:
                details.append(f"almost perfectly correlated: {correlated}")
            warnings.warn(
                f"Some parameters are not determined separately by the data "
                f"({'; '.join(details)}), so their errors are unreliable. Consider "
                "fixing one parameter of each group or adding data that "
                "distinguishes them.",
                CovarianceWarning,
                stacklevel=3,
            )

        return FitResult(
            {
                param: self._units.quantify_parameter(solution.x[i], param)
                for i, param in enumerate(self.fit_parameters)
            },
            pcov=Covariance(
                pcov,
                self.fit_parameters,
                [self._units.param_units[param] for param in self.fit_parameters],
            ),
        )

    def fit(self, p0: P0 | None = None, **kwargs) -> FitResult:
        """Run the optimization and return the fitted parameter values with
        their covariance (see FitResult).

        p0 optionally overrides initial guesses and bounds for this run only;
        it can only contain parameters given when creating the Fitter.
        Extra keyword arguments are passed to scipy.optimize.least_squares.

        Warns with ToleranceWarning if the solver tolerances are too loose for
        the default finite-difference step (not checked if diff_step is given).
        """
        if "diff_step" not in kwargs:
            self._check_tolerances()

        if p0 is None:
            x0, bounds = self._x0, self._bounds
        else:
            unknown = [str(p) for p in p0 if p not in self._units.param_units]
            if unknown:
                raise ValueError(f"{unknown} are not parameters of this Fitter")
            clean_p0, bounds_p0 = parse_p0(p0)
            x0 = self._make_x0({**self._clean_p0, **clean_p0})
            bounds = self._make_bounds({**self._bounds_p0, **bounds_p0})

        solution = least_squares(self.f, x0, bounds=bounds, **kwargs)
        return self._make_result(solution, bounds)


def parse_p0(
    p0: Mapping[Components, None | Initial | Sequence[Initial]],
) -> tuple[Mapping[Components, Initial], Mapping[Components, tuple[Initial, Initial]]]:
    clean_p0 = {}
    bounds = {}
    is_tuple = False
    for param, value in p0.items():
        try:
            n = len(value)
            is_tuple = True
        except TypeError:
            is_tuple = False

        if is_tuple:
            if n != 3:
                raise TypeError(
                    "p0 values must be None, an initial condition or a tuple (initial condition, lower bound, upper bound)"
                )
            if value[0] is not None:
                clean_p0[param] = value[0]
            bounds[param] = (value[1], value[2])
        elif value is not None:
            clean_p0[param] = value
    return clean_p0, bounds


def make_target(
    results: Mapping[Components | str, Sequence | pint.Quantity],
    save_at: Sequence | pint.Quantity,
) -> xr.Dataset:
    ureg = pint.get_application_registry()

    ureg.force_ndarray_like = True

    try:
        data_vars = {}

        if isinstance(save_at, pint.Quantity):
            units_map = {"time": save_at.units}
            dequantified_save_at = save_at.magnitude

        else:
            units_map = {}
            dequantified_save_at = save_at

        for key, value in results.items():
            if isinstance(value, pint.Quantity):
                units_map[key] = value.units
                dequantified_value = value.magnitude
            else:
                dequantified_value = value

            data_vars[key] = xr.DataArray(
                dequantified_value,
                dims=["time"],
            )

        ds = xr.Dataset(data_vars=data_vars, coords={"time": dequantified_save_at})

        ds = ds.pint.quantify(units_map)

    finally:
        ureg.force_ndarray_like = False

    return ds

