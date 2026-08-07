import numpy as np
from pytest import mark

from ... import Parameter, System, Variable, solvers
from ...simulator import Backend, Simulator

SOLVE_TOLERANCE = 1e-6
TEST_TOLERANCES = 1e-3


class Oscillator(System):
    x = Variable(initial=1)
    vx = x.derive(initial=0)
    spring_constant = Parameter(default=1)
    spring = vx.derive() << -spring_constant * x


def solution(t, x=1):
    return x * np.cos(t)


@mark.parametrize("backend", Backend.__args__)
def test_backends(backend):
    t = np.linspace(0, 10, 100)

    sim = Simulator(Oscillator, backend=backend)
    result = sim.with_solver(
        solvers.LSODA(atol=SOLVE_TOLERANCE, rtol=SOLVE_TOLERANCE)
    ).solve(
        save_at=t,
    )
    assert np.allclose(
        result["x"],
        solution(t),
        atol=TEST_TOLERANCES,
        rtol=TEST_TOLERANCES,
    )


@mark.parametrize("solver", [getattr(solvers, k) for k in solvers.__all__])
def test_solvers(solver):
    t = np.linspace(0, 10, 100)

    sim = Simulator(Oscillator)
    result = sim.with_solver(solver(atol=SOLVE_TOLERANCE, rtol=SOLVE_TOLERANCE)).solve(
        save_at=t,
    )
    assert np.allclose(
        result["x"],
        solution(t),
        atol=TEST_TOLERANCES,
        rtol=TEST_TOLERANCES,
    )
