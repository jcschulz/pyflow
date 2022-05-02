import numpy as np
from typing import Optional

class StegerWarmingMethod:
    def __init__(self, Problem, CFL: float = 0.5):

        self.Model = Problem.model()

        self.x, self.xc, self.area = Problem.geometry()

        self.apply_bcs = Problem.boundary_conditions()

        self.V = Problem.initialize_primitives()
        self.U = self.Model.primitives_to_conservatives(self.V)

        self.CFL = CFL

    def solve(self, t_final:float):
        face_area = 0.5 * (self.area[1:] + self.area[:-1])
        dx_min = (self.x[1:] - self.x[:-1]).min()

        # Quasi-1D source term / Non-zero except for momentum in cases of area change
        S = np.zeros_like(self.U)

        time = 0
        iteration = 0
        while time < t_final:
            dt = self.CFL * dx_min / self.Model.max_characteristic(self.V)
            # dt = 2.0 * dx_min / (self.V[1,0] + self.Model.Thermodynamics.speed_of_sound(self.V[2,0], self.V[0,0]))

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            F_min, F_pls = self.Model.split_flux(self.V)

            F_r = (F_pls[:,1:-1] + F_min[:,2:]) * self.area[1:]
            F_l = (F_pls[:,:-2] + F_min[:,1:-1]) * self.area[:-1]

            # Add the quasi-1D source term -- the pressure force from the area change driving
            # the flow through the nozzle.
            #
            S[self.Model.MOMENTUM,1:-1] = dt_dx * self.V[self.Model.PRESSURE,1:-1] * (self.area[1:] - self.area[:-1]) / face_area

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F_r - F_l) / face_area + S[:,1:-1]
            self.U[:,-1] = self.apply_bcs(dt_dx, self.V)

            self.V = self.Model.conservatives_to_primitives(self.U)

            iteration += 1

        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')

    def get_solution(self):
        return self.Model.solution_vector(self.xc, self.V)

    @property
    def mach_number(self):
        return self.V[self.Model.SPEED] / self.Model.Thermodynamics.speed_of_sound(self.V[self.Model.PRESSURE], self.V[self.Model.DENSITY])


class MacCormackMethod:
    def __init__(self, Problem, CFL:float = 0.5, dissipation:float = 0.0):

        self.Model = Problem.model()

        self.x, self.xc, self.area = Problem.geometry()

        self.apply_bcs = Problem.boundary_conditions()

        self.V = Problem.initialize_primitives()
        self.U = self.Model.primitives_to_conservatives(self.V)

        self.eps = dissipation
        self.CFL = CFL

    def solve(self, t_final:float):
        face_area = 0.5 * (self.area[1:] + self.area[:-1])
        dx_min = (self.x[1:] - self.x[:-1]).min()

        # Quasi-1D source term / Non-zero except for momentum in cases of area change
        S = np.zeros_like(self.U)

        # Storage for the predictor step
        U_predictor = np.copy(self.U)

        time = 0.0
        iteration = 0
        while time < t_final:

            dt = self.CFL * dx_min / self.Model.max_characteristic(self.V)
            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / dx_min

            # Compute Euler fluxes / Predictor step
            F = self.Model.flux(self.V[:,1:]) * self.area

            # Add MacCormack dissipation
            V_face = 0.5 * (self.V[:,:-1] + self.V[:,1:])
            F_eps = self.eps * self.Model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1]) * self.area
            F -= F_eps

            U_predictor[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1]) / face_area
            U_predictor[:,-1] = self.apply_bcs(dt_dx, self.V)

            V = self.Model.conservatives_to_primitives(U_predictor)

            # Compute Euler fluxes / Corrector step
            F = self.Model.flux(V[:,:-1]) * self.area

            # Add MacCormack dissipation
            V_face = 0.5 * (V[:,:-1] + V[:,1:])
            F_eps = self.eps * self.Model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1]) * self.area
            F -= F_eps

            # Add the quasi-1D source term -- the pressure force from the area change driving
            # the flow through the nozzle.
            #
            S[self.Model.MOMENTUM,1:-1] = dt_dx * self.V[self.Model.PRESSURE,1:-1] * (self.area[1:] - self.area[:-1]) / face_area

            self.U[:,1:-1] = 0.5 * (U_predictor[:,1:-1] + self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1]) / face_area) + S[:,1:-1]

            self.U[:,-1] = self.apply_bcs(dt_dx, self.V)

            if np.any(np.isnan(self.U)):
                print(f'Solution failed to converge. Exiting.')
                break
            self.V = self.Model.conservatives_to_primitives(self.U)

            iteration += 1

        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')


    def get_solution(self):
        return self.Model.solution_vector(self.xc, self.V)
