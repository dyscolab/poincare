import numpy as np

from ... import (
    Parameter,
    Simulator,
    SteadyState,
    System,
    Variable,
    assign,
    initial,
)


class SingleStable(System):
    x: Variable = initial(default=-1)

    r: Parameter = assign(default=1)

    eq = x.derive() << -(x - r)


class Bistable(System):
    x: Variable = initial(default=-1)

    r: Parameter = assign(default=1)

    eq = x.derive() << r * x - x**3


rtol = 0.01
atol = 0.1


def test_sweep():
    sim = Simulator(SingleStable)
    steady = SteadyState()
    r_values = np.arange(-5, 5.1, 1)
    result = steady.sweep(
        sim,
        variable=SingleStable.r,
        values=r_values,
    )["x"].to_numpy()
    comparison = np.abs(result - r_values) < np.maximum(
        rtol * (np.abs(result) + np.abs(r_values)) / 2, atol
    )
    assert np.all(comparison)


def test_sweep_up_down():
    sim = Simulator(Bistable)
    steady = SteadyState()
    r_values = np.arange(-5, 5.1, 1)
    simulation = steady.sweep_up_and_down(
        sim,
        variable=Bistable.r,
        values=r_values,
    )
    result = simulation["x"].to_numpy()
    doub_rev_r = np.array([r_values, np.flip(r_values)]).T
    comparison = (
        np.minimum.reduce(
            np.array(
                [
                    np.abs(result - doub_rev_r),
                    np.abs(result + doub_rev_r),
                    np.abs(result),
                ]
            )
        )
        < np.maximum.reduce(
            np.array(
                [
                    (np.abs(result) + np.abs(doub_rev_r)) / 2,
                    np.abs(result),
                    np.full_like(result, atol),
                ]
            )
        )
    )  # Compare difference to all equilibriums (+- r, 0), to atol and relative differences
    assert np.all(comparison)


def test_bistability():
    sim = Simulator(Bistable)
    steady = SteadyState()
    r_values = np.arange(-5, 5.1, 1)
    steady.bistability(
        sim,
        variable=Bistable.r,
        values=r_values,
    )
