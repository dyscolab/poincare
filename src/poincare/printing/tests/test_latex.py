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
from ...units import ureg
from ..latex import ToLatex, model_report, parameter_table, variable_table
from ..notebook_display import default_notebook_sections, notebook_report


class Oscillator(System):
    t = Independent()
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)
    phase: Parameter = assign(default=0)
    F: Parameter = assign(default=real.cos(t + phase))
    spring = vx.derive() << -x + F


# @mark.xfail(reason="Not updated to symbolite 1.0.0")
def test_model_report():
    output = model_report(Oscillator)
    assert "phase" in output


# @mark.xfail(reason="Not updated to symbolite 1.0.0")
def test_model_report_transform():
    output = model_report(Oscillator, transform={Oscillator.phase: "\\phi"})
    assert "\\phi" in output
    assert "phase" not in output


# @mark.xfail(reason="Not updated to symbolite 1.0.0")
def test_replace_algebraics():
    output = model_report(Oscillator, replace_algebraics=True)
    assert "+ F" not in output
    assert "+ \\cos" in output


def test_notebook_report():
    output = notebook_report(Oscillator, show=False)
    for title in default_notebook_sections:
        assert f"## {title}" in output
    assert "\\[" not in output
    assert "$$" in output


def test_notebook_report_sections():
    output = notebook_report(
        Oscillator,
        transform={Oscillator.phase: "\\phi"},
        sections={"Custom": lambda model, latex: latex.transform[model.phase]},
        level=3,
        show=False,
    )
    assert output == "### Custom\n\n\\phi"


class Decay(System):
    k: Parameter = assign(default=3 / ureg.s)
    f: Parameter = assign(default=2 * k)
    g: Parameter = assign(default=k + 1 / ureg.s)
    x: Variable = initial(default=2 * ureg.mol)
    vx: Derivative = x.derive(initial=0.5 * ureg.mol / ureg.s)
    eq = vx.derive() << -(f + g + k) * vx


def test_parameter_table_values():
    rows = dict(ToLatex(Decay, transform={Decay.k: "k_1"}).yield_parameters())
    assert rows["k_{1}"] == "3\\ \\frac{1}{\\mathrm{s}}"
    assert rows["f"] == "2 \\cdot k_{1}"
    assert rows["g"] == "k_{1} + 1\\ \\frac{1}{\\mathrm{s}}"


def test_variable_table_values():
    rows = {name: default for name, default, _ in ToLatex(Decay).yield_variables()}
    assert rows["x"] == "2\\ \\mathrm{mol}"
    assert rows["vx"] == "0.5\\ \\frac{\\mathrm{mol}}{\\mathrm{s}}"
    assert "Real#" not in variable_table(Decay) + parameter_table(Decay)
