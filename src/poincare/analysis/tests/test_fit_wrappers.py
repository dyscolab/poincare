import numpy as np
import pint
import xarray as xr

from ... import (
    Derivative,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
    solvers,
)
from .. import SteadyState, fit_result, fit_steady_state, make_target
from ..fits import Fitter

ureg = pint.get_application_registry()
tight = solvers.LSODA(rtol=1e-10, atol=1e-12)
rtol = 0.025


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


class Turnover(System):
    x: Variable = initial(default=0)
    k_in: Parameter = assign(default=1)
    k_out: Parameter = assign(default=1)

    eq = x.derive() << k_in - k_out * x


t = np.linspace(0, 10, 100)


def test_fit_result():
    sim = Simulator(Oscillator).with_solver(tight)
    target = make_target(
        results={"x": np.cos(2 * t), Oscillator.vx: -2 * np.sin(2 * t)}, save_at=t
    )
    fit = fit_result(
        sim, target, p0={Oscillator.omega: 2.2, Oscillator.x: (None, -np.inf, 10)}
    )
    assert np.isclose(fit[Oscillator.omega], 2, rtol=rtol)
    assert np.isclose(fit[Oscillator.x], 1, rtol=rtol)


def test_fit_result_with_units():
    sim = Simulator(UnitsOscillator).with_solver(tight)
    u_t = t * ureg.s
    target = make_target(results={"x": np.cos(2 * t) * 100 * ureg.cm}, save_at=u_t)
    fit = fit_result(sim, target, p0={UnitsOscillator.omega: 2.2 / ureg.s})
    assert np.isclose(fit[UnitsOscillator.omega].to(1 / ureg.s).magnitude, 2, rtol=rtol)


def test_fit_result_scale_and_sigma():
    """Same as the equivalent Fitter."""
    sim = Simulator(Oscillator).with_solver(tight)
    rng = np.random.default_rng(0)
    target = make_target(
        results={
            "x": np.cos(2 * t) + rng.normal(0, 0.05, t.size),
            "vx": -2 * np.sin(2 * t) + rng.normal(0, 0.1, t.size),
        },
        save_at=t,
    )
    sigma = make_target(results={"x": 0.05 * np.ones_like(t)}, save_at=t)
    p0 = {Oscillator.omega: 2.2}
    scale = {"vx": 0.5}

    fit = fit_result(sim, target, p0=p0, scale=scale, sigma=sigma)
    expected = Fitter(
        sim,
        [lambda sim: sim.solve(save_at=t)],
        [target],
        p0=p0,
        scales=scale,
        sigma=[sigma],
    ).fit()
    assert np.isclose(fit[Oscillator.omega], expected[Oscillator.omega])
    assert np.allclose(np.asarray(fit.pcov), np.asarray(expected.pcov))


def test_fit_steady_state():
    sim = Simulator(Turnover).with_solver(tight)
    k_in = np.array([0.5, 1.0, 2.0, 4.0])
    k_out_real = 0.5
    # x_ss = k_in / k_out
    target = xr.Dataset(
        {"x": xr.DataArray(k_in / k_out_real, dims="k_in", coords={"k_in": k_in})}
    )
    steady = SteadyState()
    fit = fit_steady_state(
        sim, steady, Turnover.k_in, target, p0={Turnover.k_out: (1.0, 1e-3, 10)}
    )
    # SmallDerivatives stops slightly before the exact steady state
    assert np.isclose(fit[Turnover.k_out], k_out_real, rtol=rtol)

    # DataArray target with sigma
    fit = fit_steady_state(
        sim,
        steady,
        Turnover.k_in,
        target["x"],
        p0={Turnover.k_out: (1.0, 1e-3, 10)},
        sigma=0.01 * target["x"],
    )
    assert np.isclose(fit[Turnover.k_out], k_out_real, rtol=rtol)
