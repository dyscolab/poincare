# from .. import Simulator
# from ..types import Parameter, System, Variable, assign, initial


# def test_negative_params():
#     class Model(System):
#         x: Variable = initial(default=1)

#         k: Parameter = assign(default=-1)

#         eq = x.derive() << k * x

#     sim = Simulator(Model)
#     t = range(3)
#     sim.interact(save_at=t)
