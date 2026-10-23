import warnings

import numpy as np
import pint
import pytest
from scipy.optimize import curve_fit

from ... import (
    Derivative,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
)
from ... import solvers
from ..fits import CovarianceWarning, Fitter, FitResult, ToleranceWarning, make_target

ureg = pint.get_application_registry()


class Oscillator(System):
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)

    omega: Parameter = assign(default=1)

    spring = vx.derive() << -(omega**2) * x


class UnitsOscillator(System):
    x: Variable = initial(default=1 * ureg.m)
    vx: Derivative = x.derive(initial=0 * ureg.m / ureg.s)

    omega: Parameter = assign(default=1 * 1 / ureg.s)

    spring = vx.derive() << -(omega**2) * x


t = np.linspace(0, 10, 100)
omega_real = 2
rtol = 0.025
tight = solvers.LSODA(rtol=1e-10, atol=1e-12)


def solve_at(save_at):
    return lambda sim: sim.solve(save_at=save_at)


def test_fit_without_units():
    sim = Simulator(Oscillator).with_solver(tight)
    target = make_target(
        results={
            "x": np.cos(omega_real * t),
            Oscillator.vx: -omega_real * np.sin(omega_real * t),
        },
        save_at=t,
    )

    with pytest.raises(pint.PintError):
        Fitter(
            sim,
            [solve_at(t)],
            [target],
            p0={
                Oscillator.omega: omega_real * 1.1,
                Oscillator.x: (None, -np.inf * ureg.m, 10),
            },
        )

    fitter = Fitter(
        sim,
        [solve_at(t)],
        [target],
        p0={Oscillator.omega: omega_real * 1.1, Oscillator.x: (None, -np.inf, 10)},
    )
    fit = fitter.fit()

    assert np.isclose(fit[Oscillator.x], Oscillator.x.initial, rtol=rtol)
    assert np.isclose(fit[Oscillator.omega], omega_real, rtol=rtol)


def test_fit_with_units():
    usim = Simulator(UnitsOscillator).with_solver(tight)
    u_t = t * ureg.s
    u_omega_real = 2 * 1 / ureg.s
    # Target in different (but compatible) units than the model
    u_target = make_target(
        results={
            "x": np.cos(u_omega_real * u_t) * 100 * ureg.cm,
            UnitsOscillator.vx: -u_omega_real * np.sin(u_omega_real * u_t) * ureg.m,
        },
        save_at=u_t,
    )
    fitter = Fitter(
        usim,
        [solve_at(u_t)],
        [u_target],
        p0={
            UnitsOscillator.omega: u_omega_real * 1.1,
            UnitsOscillator.x: (None, -np.inf * ureg.m, 1000 * ureg.cm),
        },
    )
    u_fit = fitter.fit()

    assert np.isclose(
        u_fit[UnitsOscillator.x].to(ureg.m).magnitude,
        UnitsOscillator.x.initial.magnitude,
        rtol=rtol,
    )
    assert np.isclose(
        u_fit[UnitsOscillator.omega].to(1 / ureg.s).magnitude,
        u_omega_real.to(1 / ureg.s).magnitude,
        rtol=rtol,
    )


def test_fit_wrong_target_units():
    usim = Simulator(UnitsOscillator).with_solver(tight)
    u_t = t * ureg.s
    target = make_target(results={"x": np.cos(omega_real * t) * ureg.s}, save_at=u_t)
    fitter = Fitter(usim, [solve_at(u_t)], [target], p0={UnitsOscillator.omega: None})
    with pytest.raises(pint.PintError):
        fitter.fit()


def test_fit_multiple_functions():
    """Two experiments with different initial conditions sharing omega."""
    sim = Simulator(Oscillator).with_solver(tight)
    t2 = np.linspace(0, 5, 30)
    x2 = 3
    target_1 = make_target(results={"x": np.cos(omega_real * t)}, save_at=t)
    # DataArray target and DataArray output
    target_2 = make_target(
        results={"x": x2 * np.cos(omega_real * t2)}, save_at=t2
    )["x"]

    fitter = Fitter(
        sim,
        [
            solve_at(t),
            lambda sim: sim.with_values({Oscillator.x: x2}).solve(save_at=t2)["x"],
        ],
        [target_1, target_2],
        p0={Oscillator.omega: (omega_real * 1.1, 0, 10)},
        scales=[{"x": 1}, 1 / x2],
    )
    fit = fitter.fit()
    assert np.isclose(fit[Oscillator.omega], omega_real, rtol=rtol)
    assert fitter.f(fitter._x0).shape == (t.size + t2.size,)


def test_dataarray_target_with_dataset_output():
    sim = Simulator(Oscillator).with_solver(tight)
    target = make_target(results={"x": np.cos(omega_real * t)}, save_at=t)["x"]
    fitter = Fitter(
        sim, [solve_at(t)], [target], p0={Oscillator.omega: omega_real * 1.1}
    )
    fit = fitter.fit()
    assert np.isclose(fit[Oscillator.omega], omega_real, rtol=rtol)


def test_fit_p0_override():
    sim = Simulator(Oscillator).with_solver(tight)
    target = make_target(results={"x": np.cos(omega_real * t)}, save_at=t)
    p0 = {Oscillator.omega: (1.5, 0, 2.5)}
    fitter = Fitter(sim, [solve_at(t)], [target], p0=p0)
    x0, bounds = fitter._x0.copy(), tuple(b.copy() for b in fitter._bounds)

    fit = fitter.fit(p0={Oscillator.omega: (2.1, 1.8, 2.2)})
    assert np.isclose(fit[Oscillator.omega], omega_real, rtol=rtol)

    # Defaults are unchanged
    assert np.array_equal(fitter._x0, x0)
    assert all(np.array_equal(a, b) for a, b in zip(fitter._bounds, bounds))
    assert dict(fitter.p0) == p0

    with pytest.raises(ValueError):
        fitter.fit(p0={Oscillator.x: 1})


def test_mismatched_lengths():
    sim = Simulator(Oscillator).with_solver(tight)
    target = make_target(results={"x": np.cos(omega_real * t)}, save_at=t)
    with pytest.raises(ValueError):
        Fitter(sim, [solve_at(t), solve_at(t)], [target])
    with pytest.raises(ValueError):
        Fitter(sim, [solve_at(t)], [target], scales=[1, 2])


def test_loose_tolerances_warn():
    target = make_target(results={"x": np.cos(omega_real * t)}, save_at=t)
    p0 = {Oscillator.omega: (2.2, 0, 10)}

    fitter = Fitter(Simulator(Oscillator), [solve_at(t)], [target], p0=p0)
    with pytest.warns(ToleranceWarning, match=r"LSODA\(rtol=1e-10, atol=1e-12\)"):
        fitter.fit()

    # Only the loose tolerance is replaced in the suggestion
    sim = Simulator(Oscillator).with_solver(solvers.LSODA(rtol=1e-12, atol=1e-6))
    fitter = Fitter(sim, [solve_at(t)], [target], p0=p0)
    with pytest.warns(ToleranceWarning, match=r"LSODA\(rtol=1e-12, atol=1e-12\)"):
        fitter.fit()

    # A custom diff_step skips the check
    with warnings.catch_warnings():
        warnings.simplefilter("error", ToleranceWarning)
        fitter.fit(diff_step=1e-4)
        Fitter(
            Simulator(Oscillator).with_solver(tight), [solve_at(t)], [target], p0=p0
        ).fit()


class Decay(System):
    x: Variable = initial(default=2.0)
    k: Parameter = assign(default=0.5)

    eq = x.derive() << -k * x


def decay_model(t, k, x0):
    return x0 * np.exp(-k * t)


t_decay = np.linspace(0, 6, 30)
sigma_decay = 0.02 + 0.05 * t_decay / 6  # different for each point
y_decay = decay_model(t_decay, 0.5, 2.0) + np.random.default_rng(1).normal(
    0, sigma_decay
)
decay_target = make_target(results={"x": y_decay}, save_at=t_decay)
decay_sigma = make_target(results={"x": sigma_decay}, save_at=t_decay)
decay_p0 = {Decay.k: 0.6, Decay.x: 1.5}


def fit_decay(**kwargs) -> FitResult:
    return Fitter(
        Simulator(Decay).with_solver(tight),
        [solve_at(t_decay)],
        [decay_target],
        p0=decay_p0,
        **kwargs,
    ).fit()


def test_covariance_matches_curve_fit():
    fit = fit_decay()
    _, pcov = curve_fit(decay_model, t_decay, y_decay, p0=[0.6, 1.5])
    assert np.allclose(np.asarray(fit.pcov), pcov, rtol=1e-4)

    fit = fit_decay(sigma=[decay_sigma])
    _, pcov = curve_fit(
        decay_model,
        t_decay,
        y_decay,
        p0=[0.6, 1.5],
        sigma=sigma_decay,
        absolute_sigma=True,
    )
    assert np.allclose(np.asarray(fit.pcov), pcov, rtol=1e-4)

    # A DataArray target with a DataArray sigma gives the same result
    fit_da = Fitter(
        Simulator(Decay).with_solver(tight),
        [solve_at(t_decay)],
        [decay_target["x"]],
        p0=decay_p0,
        sigma=[decay_sigma["x"]],
    ).fit()
    assert np.allclose(np.asarray(fit_da.pcov), pcov, rtol=1e-4)


def test_scales_do_not_change_errors_of_equivalent_fits():
    """Scaling all residuals equally changes neither the fit nor its errors."""
    fit = fit_decay(sigma=[decay_sigma])
    fit_scaled = fit_decay(sigma=[decay_sigma], scales=10.0)
    assert np.allclose(np.asarray(fit_scaled.pcov), np.asarray(fit.pcov), rtol=1e-6)

    fit = fit_decay()
    fit_scaled = fit_decay(scales={"x": 10.0})
    assert np.allclose(np.asarray(fit_scaled.pcov), np.asarray(fit.pcov), rtol=1e-6)


def test_fit_result():
    fit = fit_decay(sigma=[decay_sigma])
    assert isinstance(fit, dict)
    assert set(fit) == {Decay.k, Decay.x}
    # Can be used as values for a simulator
    Simulator(Decay).with_values(fit)

    pcov = fit.pcov
    assert pcov.parameters == [Decay.k, Decay.x]
    assert np.asarray(pcov).shape == (2, 2)
    # Indexable by pairs of parameters or names
    assert pcov["k", "x"] == pcov[Decay.k, Decay.x] == pcov["x", "k"]
    assert pcov["k", "k"] == np.asarray(pcov)[0, 0]
    with pytest.raises(KeyError):
        pcov["k", "y"]
    with pytest.raises(TypeError):
        pcov["k"]


def test_pcov_units():
    usim = Simulator(UnitsOscillator).with_solver(tight)
    u_t = t * ureg.s
    target = make_target(
        results={"x": np.cos(2 * t) * ureg.m}, save_at=u_t
    )
    fit = Fitter(
        usim,
        [solve_at(u_t)],
        [target],
        p0={UnitsOscillator.omega: 2.1 / ureg.s, UnitsOscillator.x: 1.1 * ureg.m},
    ).fit()
    assert fit.pcov["omega", "omega"].units == 1 / ureg.s**2
    assert fit.pcov["omega", "x"].units == ureg.m / ureg.s


def test_sigma_with_units():
    usim = Simulator(UnitsOscillator).with_solver(tight)
    u_t = t * ureg.s
    target = make_target(
        results={
            "x": np.cos(2 * t) * ureg.m,
            "vx": -2 * np.sin(2 * t) * ureg.m / ureg.s,
        },
        save_at=u_t,
    )
    ones = np.ones_like(t)

    def fit(sigma):
        return Fitter(
            usim,
            [solve_at(u_t)],
            [target],
            p0={UnitsOscillator.omega: 2.1 / ureg.s},
            sigma=[make_target(results=sigma, save_at=u_t)],
        ).fit()

    fit_m = fit({"x": 0.1 * ones * ureg.m, "vx": 0.2 * ones * ureg.m / ureg.s})
    fit_cm = fit(
        {"x": 10 * ones * ureg.cm, UnitsOscillator.vx: 20 * ones * ureg.cm / ureg.s}
    )
    fit_mm = fit({"x": 100 * ones * ureg.mm, "vx": 200 * ones * ureg.mm / ureg.s})
    assert np.allclose(np.asarray(fit_m.pcov), np.asarray(fit_cm.pcov))
    assert np.allclose(np.asarray(fit_m.pcov), np.asarray(fit_mm.pcov))

    # Units are not assumed: missing or different dimensionality raise
    with pytest.raises(pint.PintError):
        fit({"x": 0.1 * ones, "vx": 0.2 * ones * ureg.m / ureg.s})
    with pytest.raises(pint.PintError):
        fit({"x": 0.1 * ones * ureg.s, "vx": 0.2 * ones * ureg.m / ureg.s})


def test_sigma_parsing():
    sim = Simulator(Oscillator).with_solver(tight)
    target = make_target(
        results={"x": np.cos(omega_real * t), "vx": -2 * np.sin(omega_real * t)},
        save_at=t,
    )
    sigma_x = make_target(results={"x": 0.1 * np.ones_like(t)}, save_at=t)

    # Missing variables are padded with ones
    fitter = Fitter(sim, [solve_at(t)], [target], sigma=[sigma_x])
    assert np.array_equal(fitter._sigma[0][0], 0.1 * np.ones_like(t))
    assert np.array_equal(fitter._sigma[0][1], np.ones_like(t))

    # A target without errors is not divided by anything
    fitter = Fitter(
        sim, [solve_at(t), solve_at(t)], [target, target], sigma=[sigma_x, None]
    )
    assert fitter._sigma[1] is None
    assert fitter._weights[1] is fitter._scales[1]

    # No sigma: the residuals are only scaled
    fitter = Fitter(sim, [solve_at(t)], [target], scales={"x": 2})
    assert fitter._sigma is None
    assert fitter._weights is fitter._scales
    fitter = Fitter(sim, [solve_at(t)], [target], sigma=[None])
    assert fitter._sigma is None

    with pytest.raises(TypeError, match="list"):
        Fitter(sim, [solve_at(t)], [target], sigma=sigma_x)
    with pytest.raises(ValueError, match="targets"):
        Fitter(sim, [solve_at(t)], [target], sigma=[sigma_x, sigma_x])
    with pytest.raises(TypeError, match="Dataset"):
        Fitter(sim, [solve_at(t)], [target], sigma=[sigma_x["x"]])
    with pytest.raises(ValueError, match="not in the target"):
        Fitter(
            sim,
            [solve_at(t)],
            [target],
            sigma=[make_target(results={"y": np.ones_like(t)}, save_at=t)],
        )
    with pytest.raises(ValueError, match="shape"):
        Fitter(
            sim,
            [solve_at(t)],
            [target],
            sigma=[make_target(results={"x": np.ones(3)}, save_at=[0, 1, 2])],
        )
    with pytest.raises(ValueError, match="positive"):
        Fitter(sim, [solve_at(t)], [target], sigma=[0 * sigma_x])


def test_covariance_warnings():
    class AB(System):
        x: Variable = initial(default=2.0)
        a: Parameter = assign(default=1.0)
        b: Parameter = assign(default=0.5)

        eq = x.derive() << -a * b * x

    fitter = Fitter(
        Simulator(AB).with_solver(tight),
        [solve_at(t_decay)],
        [decay_target],
        p0={AB.a: 1.2, AB.b: 0.4, AB.x: 1.5},
    )
    with pytest.warns(CovarianceWarning, match="not determined separately"):
        fit = fitter.fit()
    # Only the product a * b is determined: the errors of a and b are huge or
    # inf (depending on whether the Jacobian is rank deficient to rounding), x is fine
    assert np.isclose(fit[AB.a] * fit[AB.b], 0.5, rtol=0.05)
    assert fit.pcov["a", "a"] > 1e6 and fit.pcov["b", "b"] > 1e6
    assert fit.pcov["x", "x"] < 1e-3

    # Exactly rank deficient: covariance set to inf instead of the pseudo-inverse's 0
    fitter = Fitter(
        Simulator(AB).with_solver(tight),
        [solve_at(t_decay)],
        [decay_target],
        p0={AB.a: 1.2, AB.b: 0.4},
    )
    with pytest.warns(CovarianceWarning, match="inf"):
        fit = fitter.fit()
    assert np.isinf(fit.pcov["a", "a"]) and np.isinf(fit.pcov["a", "b"])

    with pytest.warns(CovarianceWarning, match="bound"):
        Fitter(
            Simulator(Decay).with_solver(tight),
            [solve_at(t_decay)],
            [decay_target],
            p0={Decay.k: (0.4, 0, 0.45), Decay.x: None},  # true k is 0.5
        ).fit()
