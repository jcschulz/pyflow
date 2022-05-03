import numpy as np
from typing import Optional
from .problems import EulerQuasi1DProblem

class BaseSolver:
    def __init__(self, problem: EulerQuasi1DProblem, CFL:float = 0.5):

        self.CFL = CFL

        self.model = problem.model()
        self.x, self.xc, self.area = problem.geometry()
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

            return self.model.solution_vector(self.xc, V_normalized)
        else:
            return self.model.solution_vector(self.xc, self.V)

    def get_converstives(self):
        return self.Model.solution_vector(self.xc, self.U)

class StegerWarmingMethod(BaseSolver):

    def solve(self, t_final:float, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        # Quasi-1D source term / Non-zero except for momentum in cases of area change
        S = np.zeros_like(self.U)

        face_area = 0.5 * (self.area[1:] + self.area[:-1])
        dx_min = (self.x[1:] - self.x[:-1]).min()

        time = 0
        iteration = 0
        while time < t_final:
            dt = self.CFL * dx_min / self.model.max_characteristic(self.V)
            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            F_min, F_pls = self.model.split_flux(self.V)

            F_r = (F_pls[:,1:-1] + F_min[:,2:]) * self.area[1:]
            F_l = (F_pls[:,:-2] + F_min[:,1:-1]) * self.area[:-1]

            # Add the quasi-1D source term -- the pressure force from the area change driving
            # the flow through the nozzle.
            #
            S[self.model.MOMENTUM,1:-1] = dt_dx * self.V[self.model.PRESSURE,1:-1] * (self.area[1:] - self.area[:-1]) / face_area

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F_r - F_l) / face_area + S[:,1:-1]
            self.U[:,-1] = self.apply_bcs(dt_dx, self.V)

            if self.check():
                break

            self.V = self.model.conservatives_to_primitives(self.U)
            iteration += 1
            self.print(time, iteration, print_frequency)

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')


class MacCormackMethod(BaseSolver):

    def solve(self, t_final:float, dissipation:Optional[float] = None, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        # Quasi-1D source term / Non-zero except for momentum in cases of area change
        S = np.zeros_like(self.U)

        # Storage for the predictor step
        U_predictor = np.copy(self.U)

        face_area = 0.5 * (self.area[1:] + self.area[:-1])
        dx_min = (self.x[1:] - self.x[:-1]).min()

        time = 0.0
        iteration = 0
        while time < t_final:

            dt = self.CFL * dx_min / self.model.max_characteristic(self.V)
            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            # Compute Euler fluxes / Predictor step
            F = self.model.flux(self.V[:,1:]) * self.area

            # Add MacCormack dissipation
            V_face = 0.5 * (self.V[:,:-1] + self.V[:,1:])
            F_eps = dissipation * self.model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1]) * self.area
            F -= F_eps

            U_predictor[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1]) / face_area
            U_predictor[:,-1] = self.apply_bcs(dt_dx, self.V)

            V = self.model.conservatives_to_primitives(U_predictor)

            # Compute Euler fluxes / Corrector step
            F = self.model.flux(V[:,:-1]) * self.area

            # Add MacCormack dissipation
            V_face = 0.5 * (V[:,:-1] + V[:,1:])
            F_eps = dissipation * self.model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1]) * self.area
            F -= F_eps

            # Add the quasi-1D source term -- the pressure force from the area change driving
            # the flow through the nozzle.
            #
            S[self.model.MOMENTUM,1:-1] = dt_dx * self.V[self.model.PRESSURE,1:-1] * (self.area[1:] - self.area[:-1]) / face_area

            self.U[:,1:-1] = 0.5 * (U_predictor[:,1:-1] + self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1]) / face_area) + S[:,1:-1]
            self.U[:,-1] = self.apply_bcs(dt_dx, self.V)

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

    def solve(self, t_final:float, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        # Quasi-1D source term / Non-zero except for momentum in cases of area change
        S = np.zeros_like(self.U)

        face_area = 0.5 * (self.area[1:] + self.area[:-1])
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

            # Add the quasi-1D source term -- the pressure force from the area change driving
            # the flow through the nozzle.
            #
            S[self.model.MOMENTUM,1:-1] = dt_dx * self.V[self.model.PRESSURE,1:-1] * (self.area[1:] - self.area[:-1]) / face_area

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] * self.area[1:] - F[:,:-1] * self.area[:-1]) / face_area + S[:,1:-1]
            self.U[:,-1] = self.apply_bcs(dt_dx, self.V)

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

    def solve(self, t_final:float, print_frequency:Optional[int] = None):

        # Ensure that the solution vectors are set to their initial conditions
        self.U = self.model.primitives_to_conservatives(self.V0)
        self.V = self.model.conservatives_to_primitives(self.U)

        # Quasi-1D source term / Non-zero except for momentum in cases of area change
        S = np.zeros_like(self.U)

        face_area = 0.5 * (self.area[1:] + self.area[:-1])
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

            # Add the quasi-1D source term -- the pressure force from the area change driving
            # the flow through the nozzle.
            #
            S[self.model.MOMENTUM,1:-1] = dt_dx * self.V[self.model.PRESSURE,1:-1] * (self.area[1:] - self.area[:-1]) / face_area

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] * self.area[1:] - F[:,:-1] * self.area[:-1]) / face_area + S[:,1:-1]
            self.U[:,-1] = self.apply_bcs(dt_dx, self.V)

            if self.check():
                break

            self.V = self.model.conservatives_to_primitives(self.U)
            iteration += 1
            self.print(time, iteration, print_frequency)

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')
