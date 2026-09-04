import numpy as np
import pint
import pytest

from ... import (
    Derivative,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
)
from ..fits import fit_result, make_target

ureg = pint.get_application_registry()


def test_fit_without_units():
    class Oscillator(System):
        x: Variable = initial(default=1)
        vx: Derivative = x.derive(initial=0)

        omega: Parameter = assign(default=1)

        spring = vx.derive() << -(omega**2) * x

    sim = Simulator(Oscillator)
    t = np.linspace(0, 10, 100)
    omega_real = 2
    target_x = np.cos(omega_real * t)
    target_v = -omega_real * np.sin(omega_real * t)
    target = make_target(results={"x": target_x, Oscillator.vx: target_v}, save_at=t)

    with pytest.raises(pint.PintError):
        fit = fit_result(
            sim=sim,
            results=target,
            p0={
                Oscillator.omega: omega_real * 1.1,
                Oscillator.x: (None, -np.inf * ureg.m, 10),
            },
        )

    fit = fit_result(
        sim=sim,
        results=target,
        p0={Oscillator.omega: omega_real * 1.1, Oscillator.x: (None, -np.inf, 10)},
    )

    rtol = 0.025
    assert np.isclose(fit[Oscillator.x], Oscillator.x.initial, rtol=rtol)
    assert np.isclose(fit[Oscillator.omega], omega_real, rtol=rtol)


def test_fit_with_units():
    class UnitsOscillator(System):
        x: Variable = initial(default=1 * ureg.m)
        vx: Derivative = x.derive(initial=0 * ureg.m / ureg.s)

        omega: Parameter = assign(default=1 * 1 / ureg.s)

        spring = vx.derive() << -(omega**2) * x

    usim = Simulator(UnitsOscillator)
    u_t = np.linspace(0, 10, 100) * ureg.s
    u_omega_real = 2 * 1 / ureg.s
    u_target_x = np.cos(u_omega_real * u_t) * ureg.m
    u_target_v = -u_omega_real * np.sin(u_omega_real * u_t)
    u_target = make_target(
        results={"x": u_target_x, UnitsOscillator.vx: u_target_v}, save_at=u_t
    )
    u_fit = fit_result(
        sim=usim,
        results=u_target,
        p0={
            UnitsOscillator.omega: u_omega_real * 1.1,
            UnitsOscillator.x: (None, -np.inf * ureg.m, 1000 * ureg.cm),
        },
    )

    rtol = 0.025
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
