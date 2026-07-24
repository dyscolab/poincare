import pickle

import cloudpickle
from pytest import mark

from .. import Derivative, System, Variable, initial


class EmptyModel(System):
    pass


class SingleVariable(System):
    x: Variable = initial(default=0)


class SingleDerivative(System):
    x: Variable = initial(default=0)
    v: Derivative = x.derive(initial=0)


class SingleEquation(System):
    x: Variable = initial(default=0)
    eq = x.derive() << -x


models = [
    EmptyModel,
    SingleVariable,
    SingleDerivative,
    SingleEquation,
]


@mark.parametrize("model", models)
def test_roundtrip(model: System):
    dump = pickle.dumps(model)
    load = pickle.loads(dump)
    assert load == model


@mark.xfail(reason="cloudpickle sometimes (but not always) causes a Stack overflow.")
@mark.parametrize("model", models)
def test_local_roundtrip(model: System):
    class EmptyModel(System):
        pass

    class SingleVariable(System):
        x: Variable = initial(default=0)

    class SingleDerivative(System):
        x: Variable = initial(default=0)
        v: Derivative = x.derive(initial=0)

    class SingleEquation(System):
        x: Variable = initial(default=0)
        eq = x.derive() << -x

    inner_model = {
        EmptyModel.__name__: EmptyModel,
        SingleVariable.__name__: SingleVariable,
        SingleDerivative.__name__: SingleDerivative,
        SingleEquation.__name__: SingleEquation,
    }[model.__name__]

    dump = cloudpickle.dumps(inner_model)
    load = cloudpickle.loads(dump)
    assert load == inner_model
