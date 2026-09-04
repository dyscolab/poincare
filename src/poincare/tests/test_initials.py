from pytest import mark

from .. import Constant, Derivative, System, Variable, assign, initial


def test_variable():
    default = 0
    value = 1

    class Model(System):
        x: Variable = initial(default=default)

    assert Model.x.initial == default
    assert Model().x.initial == default
    assert Model(x=value).x.initial == value


def test_variable_with_constant():
    default = 0
    value = 1

    class Model(System):
        k: Constant = assign(default=default, constant=True)
        x: Variable = initial(default=k)

    assert Model.x.initial.default == default
    assert Model().x.initial.default == default
    assert Model(x=value).x.initial == value
    assert Model(k=value).x.initial.default == value


def test_chained_cosntants():
    class Model(System):
        k0: Constant = assign(default=0, constant=True)
        k1: Constant = assign(default=k0, constant=True)
        k2: Constant = assign(default=k1, constant=True)

    assert Model.k0.default == 0
    assert Model.k1.default == Model.k0
    assert Model.k2.default == Model.k1

    m = Model()
    assert m.k0.default == 0
    assert m.k1.default == m.k0
    assert m.k2.default == m.k1

    m = Model(k0=1)
    assert m.k0.default == 1
    assert m.k1.default == m.k0
    assert m.k2.default == m.k1


def test_derivative():
    default = 0
    value = 1

    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=default)

    assert Model.vx.initial == default
    assert Model().vx.initial == default
    assert Model(vx=value).vx.initial == value


@mark.parametrize(
    "values",
    [
        {"x": 1},
        {"vx": 1},
        {"x": 1, "vx": 1},
        {"vx": 1, "x": 1},
    ],
)
def test_variable_and_derivative(values):
    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)

    defaults = {"x": 0, "vx": 0}
    model = Model()
    for k, v in defaults.items():
        assert getattr(Model, k).initial == v
        assert getattr(model, k).initial == v

    expected = {k: values.get(k, v) for k, v in defaults.items()}
    model = Model(**values)
    for k, v in expected.items():
        assert getattr(model, k).initial == v


def test_single_composition():
    default = 0
    value = 1
    inner = 2

    class Particle(System):
        x: Variable = initial(default=inner)

    class Model(System):
        x: Variable = initial(default=default)
        p = Particle(x=x)

    assert Model.x.initial == default
    assert Model.p.x.initial == default

    assert Model().x.initial == default
    assert Model().p.x.initial == default

    assert Model(x=value).x.initial == value
    assert Model(x=value).p.x.initial == value


def test_multilevel_composition():
    defaults = [1, 2, 3]

    class First(System):
        x0: Variable = initial(default=defaults[0])

    class Second(System):
        x1: Variable = initial(default=defaults[1])
        inner = First(x0=x1)

    class Third(System):
        x2: Variable = initial(default=defaults[2])
        inner = Second(x1=x2)

    assert Third.x2.initial == defaults[2]
    assert Third.inner.x1.initial == defaults[2]
    assert Third.inner.inner.x0.initial == defaults[2]

    assert Third().x2.initial == defaults[2]
    assert Third().inner.x1.initial == defaults[2]
    assert Third().inner.inner.x0.initial == defaults[2]

    value = 0
    assert Third(x2=value).x2.initial == value
    assert Third(x2=value).inner.x1.initial == value
    assert Third(x2=value).inner.inner.x0.initial == value
