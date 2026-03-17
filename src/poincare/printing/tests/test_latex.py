from pytest import mark
from symbolite import real

from ... import (
    Derivative,
    Independent,
    Parameter,
    System,
    Variable,
    assign,
    initial,
)
from ..latex import model_report


class Oscillator(System):
    t = Independent()
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)
    phase: Parameter = assign(default=0)
    damp_rate: Parameter = assign(default=1)
    F: Parameter = assign(default=real.cos(t + phase))
    spring = vx.derive() << -x + F


@mark.xfail(reason="Not updated to symbolite 1.0.0")
def test_model_report():
    output = model_report(Oscillator)
    assert "damp_{rate}" in output


@mark.xfail(reason="Not updated to symbolite 1.0.0")
def test_model_report_transform():
    output = model_report(Oscillator, transform={Oscillator.damp_rate: "\\gamma"})
    assert "\\gamma" in output
    assert "damp_rate" not in output
    assert "damp_{rate}" not in output


@mark.xfail(reason="Not updated to symbolite 1.0.0")
def test_replace_algebraics():
    output = model_report(
        Oscillator, transform={Oscillator.damp_rate: "\\gamma"}, replace_algebraics=True
    )
    assert "+ F" not in output
    assert "+ \\cos" in output
