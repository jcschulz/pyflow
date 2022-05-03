import numpy as np
from typing import Optional

from .riemann import ExactRiemannSolver
from .problems import Euler1DProblems

class BaseSolver:
    def __init__(self, problem: Euler1DProblems, CFL:float = 0.5):

        self.CFL = CFL

        self.model = problem.model()
        self.x, self.area = problem.geometry()
        self.apply_bcs = problem.boundary_conditions()

        # Save the initial primitive state and initialize the solution vector
        self.V0 = problem.initialize_primitives()
        self.U = self.model.primitives_to_conservatives(self.V0)

    def print(self, time, iteration, frequency):
        if frequency:
            if iteration%frequency == 0:
                print(f'iteration: {iteration},  time = {time:12.5f},  p_max = {self.V[self.model.PRESSURE].max():12.5f}')

    def check(self) -> bool:
        if np.any(np.isnan(self.U)):
            print(f'Solution failed to converge. Exiting.')
            return True
        else:
            return False

    def get_primitives(self, normalize=False):
        if normalize:

            V_normalized = self.model.State(
                density = self.V[self.model.DENSITY] / self.V0[self.model.DENSITY,0],
                speed = self.V[self.model.SPEED] / self.V0[self.model.SPEED,0],
                pressure = self.V[self.model.PRESSURE] / self.V0[self.model.PRESSURE,0]
            ).stack()

            return self.model.solution_vector(self.x, V_normalized)
        else:
            return self.model.solution_vector(self.x, self.V)

    def get_converstives(self):
        return self.model.solution_vector(self.x, self.U)


class MacCormackMethod(BaseSolver):

    def solve(self, t_final:float, dissipation:Optional[float] = None, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        # Storage for the predictor step
        U_predictor = np.copy(self.U)

        dx_min = (self.x[1:] - self.x[:-1]).min()

        time = 0
        iteration = 0
        while time < t_final:
            U_predictor = np.zeros_like(self.U)

            dt = self.CFL * dx_min / self.model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            F = self.model.flux(self.V[:,1:])

            V_face = 0.5 * (self.V[:,:-1] + self.V[:,1:])
            F_eps = dissipation * self.model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1])
            F -= F_eps

            U_predictor[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(U_predictor)

            V = self.model.conservatives_to_primitives(U_predictor)
            F = self.model.flux(V[:,:-1])

            V_face = 0.5 * (V[:,:-1] + V[:,1:])
            F_eps = dissipation * self.model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1])
            F -= F_eps

            self.U[:,1:-1] = 0.5 * (U_predictor[:,1:-1] + self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1]))
            self.apply_bcs(self.U)

            if self.check():
                break

            self.V = self.model.conservatives_to_primitives(self.U)
            iteration += 1
            self.print(time, iteration, print_frequency)

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')


class StegerWarmingMethod(BaseSolver):

    def solve(self, t_final:float, dissipation:Optional[float] = None, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        dx_min = (self.x[1:] - self.x[:-1]).min()

        time = 0.0
        iteration = 0
        while time < t_final:
            dt = self.CFL * dx_min / self.model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            F_min, F_pls = self.model.split_flux(self.V)
            F_r = F_pls[:,1:-1] + F_min[:,2:]
            F_l = F_pls[:,:-2] + F_min[:,1:-1]

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F_r - F_l)
            self.apply_bcs(self.U)

            if self.check():
                break

            self.V = self.model.conservatives_to_primitives(self.U)
            iteration += 1
            self.print(time, iteration, print_frequency)

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')

class GodunovMethod(BaseSolver):

    def riemann_flux(self):
        self._nVars, self._nCells = self.V.shape

        F = np.zeros((self._nVars, self._nCells-1))
        for i in range(self._nCells-1):
            left = self.model.state_vector(self.V, i)
            right = self.model.state_vector(self.V, i+1)

            Riemann = ExactRiemannSolver(left, right, gamma=self.model.Thermodynamics.gamma)
            V = Riemann.sample(0)
            F[:,i] = self.model.flux_given_state(V)
        return F

    def solve(self, t_final:float, dissipation:Optional[float] = None, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        dx_min = (self.x[1:] - self.x[:-1]).min()

        time = 0.0
        iteration = 0
        while time < t_final:
            dt = self.CFL * dx_min / self.model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            F = self.riemann_flux()

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(self.U)

            if self.check():
                break

            self.V = self.model.conservatives_to_primitives(self.U)
            iteration += 1
            self.print(time, iteration, print_frequency)

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')

class HLLMethod(BaseSolver):

    def solve(self, t_final:float, dissipation:Optional[float] = None, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        dx_min = (self.x[1:] - self.x[:-1]).min()

        time = 0.0
        iteration = 0
        while time < t_final:
            dt = self.CFL * dx_min / self.model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            F = self.model.HLL_riemann_solver(self.V[:,:-1], self.V[:,1:])

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(self.U)

            if self.check():
                break

            self.V = self.model.conservatives_to_primitives(self.U)
            iteration += 1
            self.print(time, iteration, print_frequency)

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')


class HLLCMethod(BaseSolver):

    def solve(self, t_final:float, dissipation:Optional[float] = None, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        dx_min = (self.x[1:] - self.x[:-1]).min()

        time = 0.0
        iteration = 0
        while time < t_final:
            dt = self.CFL * dx_min / self.model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            F = self.model.HLLC_riemann_solver(self.V[:,:-1], self.V[:,1:])

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(self.U)

            if self.check():
                break

            self.V = self.model.conservatives_to_primitives(self.U)
            iteration += 1
            self.print(time, iteration, print_frequency)

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')

