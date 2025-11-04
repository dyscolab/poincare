from ..latex import model_report
from ... import Variable, Derivative, System, Constant, Parameter, initial, assign


class Oscillator(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    spring_constant: Constant = assign(default=0, constant=True)

    spring = vx.derive() << -spring_constant * x


class Dampening(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    damp_rate: Parameter = assign(default=0)

    dampening = vx.derive() << -damp_rate * vx


class DampedOscillator(System):
    # Define new external variables for the sysyem
    x_ext: Variable = initial(default=1)
    vx_ext: Derivative = x_ext.derive(initial=0)

    spring_constant: Constant = assign(default=1, constant=True)
    damp_rate: Parameter = assign(default=0.1)

    # Apply the models to the external systems variables
    oscillator = Oscillator(x=x_ext, spring_constant=spring_constant)
    dampening = Dampening(x=x_ext, damp_rate=damp_rate)


def test_model_report():
    output = model_report(DampedOscillator)

def test_model_report_transform():

