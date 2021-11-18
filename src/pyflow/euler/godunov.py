
import numpy as np

from pyflow.euler.state import ConservedState, PrimitiveState
from pyflow.euler.exact_riemann import ExactRiemannSolver

class Godunov():

    __StateVariables = 3
    __DENSITY = 0
    __MOMENTUM = 1
    __ENERGY = 2

    __NumberOfGhosts = 1

    def __init__(self, Lx, Nx, time, gamma=1.4, CFL=0.9):

        self.x = np.linspace(0, Lx, Nx)
        self.dx = self.x[1] - self.x[0]
        self.time = 0
        self.final_time = time
        self.CFL = CFL

        self.gamma = gamma

        # Increase the size of the solution domain to account for ghost cells
        #
        self.__L = self.__NumberOfGhosts
        self.__R = -self.__NumberOfGhosts
        self.U = np.zeros((self.__StateVariables,Nx+2*self.__NumberOfGhosts))

        self.Riemann = ExactRiemannSolver(gamma=gamma)

    def primitive_state_at_face(self, CellCenter_U):
        return PrimitiveState(
            rho = CellCenter_U[self.__DENSITY],
            u = CellCenter_U[self.__MOMENTUM] / CellCenter_U[self._DENSITY],
            p = self.compute_pressure(
                CellCenter_U[self.__ENERGY], CellCenter_U[self._DENSITY],
                CellCenter_U[self._MOMENTUM] / CellCenter_U[self._DENSITY]
            )
        )

    def compute_timestep(self):
        fastest_wave = np.abs(self.velocity + self.speed_of_sound)
        return self.CFL * self.dx / np.max(fastest_wave)

    def solve(self):
        while self.time - self.final_time > 0:

            dt = self.compute_timestep()
            if self.time + dt > self.final_time:
                dt = self.final_time - self.time
            self.time += dt

            self.explicit_euler(dt)

    def explicit_euler(self, dt):

        self.boundary_conditions()

        F_left, F_right = self.compute_flux()

        self.U[:,self.__L:self.__R] -= dt * (F_right - F_left) / self.dx

    def compute_flux(self):

        # Cells   i-1   |    i    |   i+1   |
        #               |         |         |
        # Faces         j        j+1       j+2
        #
        # Riemann Problem at Face j uses v_L(i-1) and v_M(i)
        # Riemann Problem at Face j+1 uses v_M(i) and v_R(i+1)

        v_L = self.primitive_state_at_face(self.U[:,:self.__R-1])
        v_M = self.primitive_state_at_face(self.U[:,1:self.__R])
        v_R = self.primitive_state_at_face(self.U[:,2:])

        # Compute the fluxes
        #
        return self.Riemann.flux(v_L, v_M), self.Riemann.flux(v_M, v_R)

    def boundary_conditions(self):

        # Transmissive boundary conditions at x=0 and x=Lx
        self.U[0] = self.U[1]
        self.U[-1] = self.U[-2]

    @property
    def density(self):
        return self.U[self.__DENSITY,self.__L:self.__R]

    @property
    def momentum(self):
        return self.U[self.__MOMENTUM,self.__L:self.__R]

    @property
    def energy(self):
        return self.U[self.__ENERGY,self.__L:self.__R] / self.U[self.__DENSITY,self.__L:self.__R] - 0.5 * self.velocity**2

    @property
    def velocity(self):
        return self.U[self.__MOMENTUM,self.__L:self.__R] / self.U[self.__DENSITY,self.__L:self.__R]

    @property
    def pressure(self):
        return self.U[self.__DENSITY,self.__L:self.__R] * self.energy * (self.gamma - 1)

    @property
    def speed_of_sound(self):
        return np.sqrt(self.energy * self.gamma * (self.gamma - 1))

    def compute_internalenergy(self, pressure, density):
        return pressure / density / (self.gamma - 1.0)

    def compute_pressure(self, total_energy, density, u):
        return (total_energy - 0.5 * density * u**2 ) / (self.gamma - 1)

    def conservatives(self, V: PrimitiveState) -> ConservedState:
        return ConservedState(
            rho = V.rho,
            rhoU = V.rho * V.u,
            E = V.rho * (self._internalenergy(V.p, V.rho) + 0.5 * V.u**2)
        )

    def primitives(self, Q : ConservedState) -> PrimitiveState:
        return PrimitiveState(
            rho = Q.rho,
            u = Q.rhoU * Q.rho,
            p = self.pressure(Q.rho, Q.E)
        )

    def initialize_shocktube(self, diaphragm, left : PrimitiveState, right : PrimitiveState):

        i_diaphragm = next(i for i,x in enumerate(self.x) if x > diaphragm)

        Conservatives = self.conservatives(left)
        self.U[self._DENSITY, :i_diaphragm] = Conservatives.rho
        self.U[self._MOMENTUM,:i_diaphragm] = Conservatives.rhoU
        self.U[self._ENERGY,  :i_diaphragm] = Conservatives.E

        Conservatives = self.conservatives(right)
        self.U[self._DENSITY, i_diaphragm:] = Conservatives.rho
        self.U[self._MOMENTUM,i_diaphragm:] = Conservatives.rhoU
        self.U[self._ENERGY,  i_diaphragm:] = Conservatives.E





