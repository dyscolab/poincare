import matplotlib.pyplot as plt
import numpy as np

from poincare import Parameter, Simulator, System, Variable, assign, initial


class LotkaVolterra(System):
    prey: Variable = initial(default=10)
    predator: Variable = initial(default=1)

    prey_birth_rate: Parameter = assign(default=1)
    prey_death_rate: Parameter = assign(default=1)
    predator_death_rate: Parameter = assign(default=1)
    predator_birth_rate: Parameter = assign(default=1)

    birth_prey = prey.derive() << prey_birth_rate * prey
    death_prey = prey.derive() << -prey_death_rate * prey * predator

    birth_predator = predator.derive() << predator_birth_rate * prey * predator
    death_predator = predator.derive() << -predator_death_rate * predator


if __name__ == "__main__":
    sim = Simulator(LotkaVolterra)
    result = sim.solve(save_at=np.linspace(0, 100, 1000))
    result.to_dataframe().plot()
    plt.show()
