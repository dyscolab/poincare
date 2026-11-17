import numpy as np
import pint
import xarray as xr
from pytest import raises

from ... import solvers
from ...simulator import Simulator
from ...types import Independent, Parameter, System, Variable, assign, initial

u = pint.get_application_registry()


class Decay(System):
    x: Variable = initial(default=1)
    k: Parameter = assign(default=1)
    eq = x.derive() << -k * x


class DecayUnits(System):
    t = Independent(default=0 * u.s)
    x: Variable = initial(default=1)
    k: Parameter = assign(default=1 / u.s)
    eq = x.derive() << -k * x


def test_with_times_matches_solve():
    save_at = np.linspace(0, 2, 11)
    sim = Simulator(Decay).with_solver(solvers.LSODA(atol=1e-9, rtol=1e-9))
    expected = sim.solve(save_at=save_at)
    result = sim.with_times(save_at=save_at).solve()
    xr.testing.assert_identical(result, expected)
    np.testing.assert_allclose(result.x, np.exp(-save_at), rtol=1e-6)

    t_span = (-1, 2)
    expected = sim.solve(t_span=t_span, save_at=save_at)
    result = sim.with_times(t_span=t_span, save_at=save_at).solve()
    xr.testing.assert_identical(result, expected)


def test_with_times_is_immutable_and_propagates():
    sim = Simulator(Decay)
    timed = sim.with_times(save_at=[0, 1, 2])
    assert sim.save_at is None and sim.t_span is None

    with raises(TypeError, match="t_span and/or save_at"):
        sim.solve()

    for derived in (
        timed.with_values({Decay.k: 2}),
        timed.with_solver(solvers.RK45()),
        timed.with_transform([Decay.x]),
    ):
        assert list(derived.solve().time.values) == [0, 1, 2]

    # with_times keeps values set before it
    result = (
        sim.with_values({Decay.k: 2})
        .with_solver(solvers.LSODA(atol=1e-6, rtol=1e-6))
        .with_times(save_at=[0, 1])
        .solve()
    )
    np.testing.assert_allclose(result.x[-1], np.exp(-2), rtol=1e-3)


def test_with_times_requires_an_argument():
    with raises(TypeError):
        Simulator(Decay).with_times()


def test_with_times_units():
    sim = Simulator(DecayUnits)
    result = sim.with_times(save_at=np.array([0, 1000]) * u.ms).solve()
    assert result.time.pint.units == u.ms

    with raises(pint.PintError):
        sim.with_times(save_at=[0, 1]).solve()
