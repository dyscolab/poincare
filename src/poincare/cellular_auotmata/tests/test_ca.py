import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from collections import Counter

from ..ca_simulator import CASimulator
from ..ca_types import Automata, SquareAutomata, Square, create_cell_interaction
from ...types import System, Variable, Derivative, Parameter, initial, assign
from ...compile import build_first_order_symbolic_ode


class Flow(System):
    w: Variable = initial(default=1)
    o: Variable = initial(default=0.5)

    k: Parameter = assign(default=1)

    eq_1 = w.derive() << 0 * k
    eq_2 = o.derive() << 0


def flow_interaction(cls, int: Flow, ext: Flow):
    cls.eq_1 = int.w.derive() << int.k * (ext.w - int.w)
    cls.eq_2 = int.o.derive() << int.k * (ext.o - int.o)


class Diffusion(System):
    w: Variable = initial(default=1)

    k: Parameter = assign(default=1)

    eq_1 = w.derive() << 0 * k


def diffusion_interaction(cls, int: Diffusion, ext: Diffusion):
    cls.eq_1 = int.w.derive() << int.k * (ext.w - int.w)


def diffusion_boundary(cls, int: Flow):
    # cls.eq_1 = int.w.derive() << int.k * (0 - int.w)
    pass


def test_create_interaction_system():
    created_system = create_cell_interaction(interaction=flow_interaction, cell=Flow)
    symbolic = build_first_order_symbolic_ode(created_system)
    assert Counter([str(var) for var in symbolic.variables]) == Counter(
        ["int_o", "out_o", "int_w", "out_w"]
    )
    assert Counter([str(param) for param in symbolic.parameters]) == Counter(
        ["int_k", "out_k"]
    )


def animate_diffusion():
    shape = (30, 30)
    save_at = np.arange(0, 10, 1)
    D = 1
    automata = SquareAutomata(
        cell=Diffusion,
        cell_interaction=diffusion_interaction,
        boundary=diffusion_boundary,
        geometry=Square,
    )
    ca_sim = CASimulator(model=automata, shape=shape)
    w_initial = np.zeros(shape)

    # set centre to intensity equal to number of cells
    w_initial[int((shape[0] - 1) / 2), int((shape[1] - 1) / 2)] = np.prod(shape)

    ca_sim.solve_and_animate(
        save_at=save_at,
        values={Diffusion.w: w_initial, Diffusion.k: np.full_like(w_initial, D)},
        variable=Flow.w,
        save_to="animation_tests/diffusion_animation.gif",
    )


def test_diffusion():
    shape = (30, 30)
    save_at = np.arange(0, 10, 1)
    D = 1
    automata = SquareAutomata(
        cell=Diffusion,
        cell_interaction=diffusion_interaction,
        boundary=diffusion_boundary,
        geometry=Square,
    )
    ca_sim = CASimulator(model=automata, shape=shape)
    w_initial = np.zeros(shape)

    # set centre to intensity equal to number of cells
    w_initial[int((shape[0] - 1) / 2), int((shape[1] - 1) / 2)] = np.prod(shape)

    result = ca_sim.solve(
        save_at=save_at,
        values={Diffusion.w: w_initial, Diffusion.k: np.full_like(w_initial, D)},
        format="array",
    )

    def gaussian_2d(coords, cov_xx, cov_xy, cov_yx, cov_yy, mu_x, mu_y):
        x, y = coords  # both are 1D arrays of length N

        mu = np.array([mu_x, mu_y])
        Sigma = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
        try:
            invSigma = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            return 1000

        # Stack coords into (2, N)
        XY = np.vstack([x, y])

        # (XY - mu.reshape(2,1)) → shape (2, N)
        D = XY - mu.reshape(2, 1)
        exponent = -0.5 * np.abs(np.sum(D * (invSigma @ D), axis=0))
        return np.exp(exponent)

    # Example fitting loop
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xdata = (x.ravel(), y.ravel())

    covariances = np.empty((result.shape[0], 4))

    for i in range(result.shape[0]):
        zdata = result[i].ravel() / np.max(result[i])
        # initial guesses
        p0 = [
            4 * D * save_at[i],
            0,
            0,
            4 * D * save_at[i],
            x.mean(),
            y.mean(),
        ]

        popt, pcov = curve_fit(gaussian_2d, xdata, zdata, p0=p0)
        covariances[i] = popt[:4]

    # print(covariances)
    plt.plot(save_at, covariances.T[0], label="x variance")
    plt.plot(save_at, covariances.T[1], label="xy covariance")
    plt.plot(save_at, covariances.T[2], label="yx covariance")
    plt.plot(save_at, covariances.T[3], label="y variance")
    plt.plot(save_at, np.array([2 * D * t for t in save_at]), label="2Dt")
    plt.legend()
    # plt.show()
    assert True
