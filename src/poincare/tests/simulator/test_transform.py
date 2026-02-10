import numpy as np
from pytest import mark

from ... import Constant, Parameter, System, Variable
from ...simulator import Simulator
from ...types import Independent


class Model(System):
    time = Independent(default=0)
    c = Constant(default=0)
    unused = Constant(default=0)
    x = Variable(initial=c)
    y = Variable(initial=0)
    k = Parameter(default=1)
    F = Parameter(default=time)

    eq_x = x.derive() << k * x
    eq_y = y.derive() << F


times = np.linspace(0, 10, 100)


def test_one_variable():
    ds_all = Simulator(Model).solve(save_at=times)
    ds = Simulator(Model, transform={"x": Model.x}).solve(save_at=times)

    assert ds.sizes["time"] == ds_all.sizes["time"]
    assert (ds["x"].values == ds_all["x"].values).all()
    assert set(ds.data_vars) == {"x"}


def test_sum_variable():
    ds_all = Simulator(Model).solve(save_at=times)
    ds = Simulator(Model, transform={"sum": Model.x + Model.y}).solve(save_at=times)

    assert ds.sizes["time"] == ds_all.sizes["time"]
    assert (ds["sum"].values == (ds_all["x"].values + ds_all["y"].values)).all()
    assert set(ds.data_vars) == {"sum"}


@mark.xfail(reason="not implemented")
def test_non_variable():
    # Should it shortcut and skip the solver?
    sim = Simulator(Model, transform={"c": Model.c})

    ds = sim.solve(save_at=times)
    assert np.all(ds["c"].values == Model.c.default)

    ds = sim.solve(save_at=times, values={Model.c: Model.c.default + 1})
    assert np.all(ds["c"].values == Model.c.default + 1)


def test_number():
    sim = Simulator(Model, transform={"my_number": 1})
    ds = sim.solve(save_at=times)
    assert np.all(ds["my_number"].values == 1)


@mark.xfail(reason="not implemented")
def test_unused_variable():
    sim = Simulator(Model, transform={"unused": Model.unused})

    ds = sim.solve(save_at=times)
    assert np.all(ds["unused"].values == Model.unused.default)

    ds = sim.solve(
        save_at=times,
        values={Model.unused: Model.unused.default + 1},
    )
    assert np.all(ds["unused"].values == Model.unused.default + 1)


@mark.xfail(reason="Optimization not implemented.")
def test_unused_variable_skips_solver():
    """If the transform does not need to integrate the equations, it could skip that."""
    sim = Simulator(Model, transform={"unused": Model.unused})
    ds = sim.solve(save_at=times, solver=None)
    assert np.all(ds["unused"].values == Model.unused.default)
