import numpy as np

from pyflow.euler.state import PrimitiveState
from pyflow.euler.exact_riemann import ExactRiemannSolver

class ShockTube():

    def __init__(self, Lx, Nx, time, left : PrimitiveState, right : PrimitiveState, diaphragm=0.5, gamma=1.4, CFL=0.9):

        self.x = np.linspace(0, Lx, Nx)
        self.dx = self.x[1] - self.x[0]
        self.time = time
        self.gamma = gamma
        self.density = np.zeros_like(self.x)
        self.velocity = np.zeros_like(self.x)
        self.pressure = np.zeros_like(self.x)
        self.energy = np.zeros_like(self.x)

        self.Riemann = ExactRiemannSolver(gamma=gamma)
        self.Riemann.initialize(left, right)
        self.diaphragm = diaphragm * Lx

        i_diaphragm = next(i for i,x in enumerate(self.x) if x > self.diaphragm)

        self.density[:i_diaphragm] = left.rho
        self.density[i_diaphragm:] = right.rho

        self.velocity[:i_diaphragm] = left.u
        self.velocity[i_diaphragm:] = right.u

        self.pressure[:i_diaphragm] = left.p
        self.pressure[i_diaphragm:] = right.p

    def solve(self):

        p_star, u_star = self.Riemann.star_state()

        for i in range(len(self.x)):

            s = (self.x[i] - self.diaphragm) / self.time

            state = self.Riemann.sample(p_star, u_star, s)

            self.density[i] = state.rho
            self.velocity[i] = state.u
            self.pressure[i] = state.p

    @property
    def momentum(self):
        return self.density * self.velocity

    @property
    def speed_of_sound(self):
        return np.sqrt(self.energy * self.gamma * (self.gamma - 1))
