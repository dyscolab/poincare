from .. import System, Variable, initial
from ._loop import Loop


class A(System):
    x: Variable = initial(default=0)
    r = x.derive() << x


class B(System):
    x: Variable = initial(default=0)
    y: Variable = initial(default=0)
    r_x = x.derive() << x
    r_y = y.derive() << y


def test_disconnected_loop():
    class Main(System):
        x: Variable = initial(default=0)
        r = x.derive() << 1

        loop = Loop(A())


def test_connected_loop():
    class Main(System):
        x: Variable = initial(default=0)
        r = x.derive() << 1

        loop = Loop(A(x=x))


def test_mixed_loop():
    class Main(System):
        x: Variable = initial(default=0)
        r = x.derive() << 1

        loop = Loop(B(y=x))
