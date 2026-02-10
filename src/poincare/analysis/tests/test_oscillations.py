import numpy as np
from symbolite import real

from ... import (
    Derivative,
    Independent,
    Oscillations,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
)


class ForcedDampedOscillator(System):
    t = Independent()
    # Define new external variables for the sysyem
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)

    spring_constant: Parameter = assign(default=2)
    damp_rate: Parameter = assign(default=0.3)
    T: Parameter = assign(default=0.5)
    force: Parameter = assign(default=0.2 * real.sin(2 * np.pi / T * t))

    # Apply the models to the external systems variables
    oscillator = vx.derive() << -spring_constant * x + force + 0.2
    dampening = vx.derive() << -damp_rate * vx


rtol = 0.01
atol = 0.01


def test_sweep():
    sim = Simulator(ForcedDampedOscillator)
    osc = Oscillations()
    T_values = np.arange(1, 10, 10)
    result = osc.sweep(
        sim,
        T_min=0.5,
        T_max=15,
        T_r=50,
        parameter=ForcedDampedOscillator.T,
        values=T_values,
    )
    periods = result.sel(quantity="period").to_array().values
    comparison = np.abs(periods - T_values) < np.minimum(
        rtol * (np.abs(periods) + np.max(T_values)) / 2, atol
    )
    assert np.all(comparison)


def test_instanced_system():
    sim = Simulator(ForcedDampedOscillator())
    osc = Oscillations()
    T_values = np.arange(1, 10, 10)
    result = osc.sweep(
        sim,
        T_min=0.5,
        T_max=15,
        T_r=50,
        parameter=ForcedDampedOscillator.T,
        values=T_values,
    )
    periods = result.sel(quantity="period").to_array().values
    comparison = np.abs(periods - T_values) < np.minimum(
        rtol * (np.abs(periods) + np.max(T_values)) / 2, atol
    )
    assert np.all(comparison)
