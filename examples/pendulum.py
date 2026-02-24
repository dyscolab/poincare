import matplotlib.pyplot as plt
import numpy as np
from symbolite import real

from poincare import Derivative, Parameter, Simulator, System, Variable, assign, initial


class Pendulum(System):
    angle: Variable = initial(default=0)
    angular_velocity: Derivative = angle.derive(initial=0)

    pendulum_length: Parameter = assign(default=1)
    gravity: Parameter = assign(default=9.8)

    spring = angular_velocity.derive() << -gravity / pendulum_length * real.sin(angle)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    for angle_0 in [1, 10, 30, 50, 70, 90]:
        model = Pendulum(angle=np.deg2rad(angle_0))
        result = Simulator(model).solve(save_at=np.linspace(0, 5, 1000))
        (result["angle"] / angle_0).to_dataframe().plot(ax=ax, label=angle_0)

    plt.legend(title="Angle [°]")
    plt.ylabel("Angle relative to initial angle")
    plt.show()
