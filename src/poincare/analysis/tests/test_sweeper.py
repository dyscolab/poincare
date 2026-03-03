import numpy as np

from ... import (
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
)
from ..sweeper import Sweeper


class LotkaVolterra(System):
    prey: Variable = initial(default=10)
    predator: Variable = initial(default=1)

    prey_birth_rate: Parameter = assign(default=1)
    prey_death_rate: Parameter = assign(default=1)
    predator_death_rate: Parameter = assign(default=1)
    predator_birth_rate: Parameter = assign(default=1)
    k: Parameter = assign(default=1)

    birth_prey = prey.derive() << prey_birth_rate * prey
    death_prey = prey.derive() << -prey_death_rate * prey * predator

    birth_predator = predator.derive() << predator_birth_rate * prey * predator
    death_predator = predator.derive() << -predator_death_rate * predator


sim = Simulator(LotkaVolterra)


def single_mean(ds):
    return ds["prey"].mean()


def double_mean(ds):
    return {"prey": ds["prey"].mean(), "predator": ds["predator"].mean()}


def test_single_sweep():
    sweep = Sweeper(single_mean)
    result = sweep.sweep(
        sim,
        save_at=np.linspace(0, 10, 50),
        parameter=LotkaVolterra.predator_birth_rate,
        values=np.linspace(1, 10, 50),
    )
    assert result.data_vars == np.array(["result"])
    assert np.all(result.coords["predator_birth_rate"] == np.linspace(1, 10, 50))


def test_double_sweep():
    parameter_values = np.linspace(1, 10, 50)
    sweep = Sweeper(double_mean)
    result = sweep.sweep(
        sim,
        save_at=np.linspace(0, 10, 50),
        parameter=LotkaVolterra.predator_birth_rate,
        values=parameter_values,
    )

    assert np.all(result.data_vars == np.array(["prey", "predator"]))
    assert np.all(result.coords["predator_birth_rate"] == parameter_values)


def test_other_values_in_sweep():
    parameter_values = np.linspace(1, 10, 50)
    sweep = Sweeper(double_mean)
    result = sweep.sweep(
        sim,
        save_at=np.linspace(0, 10, 50),
        parameter=LotkaVolterra.predator_birth_rate,
        values=parameter_values,
        other_values={LotkaVolterra.prey: 0, LotkaVolterra.predator: 0},
    )
    assert np.all(result["prey"] == np.zeros_like(parameter_values))
    assert np.all(result["predator"] == np.zeros_like(parameter_values))
