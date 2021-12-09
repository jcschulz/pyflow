import numpy as np

from .riemann import ExactRiemannSolver

class MacCormackMethod:
    def __init__(self, Problem, CFL: float = 0.5, dissipation: float = 0.0):

        self.Model = Problem.model()

        self.x, self.area = Problem.geometry()
        self.dx = (self.x[1:] - self.x[:-1]).min()

        self.apply_bcs = Problem.boundary_conditions()

        self.V = Problem.initialize_primitives()
        self.U = self.Model.primitives_to_conservatives(self.V)

        self.eps = dissipation
        self.CFL = CFL

    def solve(self, t_final):
        time = 0
        while time < t_final:
            U_predictor = np.zeros_like(self.U)

            dt = self.CFL * self.dx / self.Model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / self.dx

            F = self.Model.flux(self.V[:,1:])

            V_face = 0.5 * (self.V[:,:-1] + self.V[:,1:])
            F_eps = self.eps * self.Model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1])
            F -= F_eps

            U_predictor[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(U_predictor)

            V = self.Model.conservatives_to_primitives(U_predictor)
            F = self.Model.flux(V[:,:-1])

            V_face = 0.5 * (V[:,:-1] + V[:,1:])
            F_eps = self.eps * self.Model.u_plus_c(V_face) * (self.U[:,1:] - self.U[:,:-1])
            F -= F_eps

            self.U[:,1:-1] = 0.5 * (U_predictor[:,1:-1] + self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1]))
            self.apply_bcs(self.U)

            self.V = self.Model.conservatives_to_primitives(self.U)

        return self.Model.solution_vector(self.x, self.V)


class StegerWarmingMethod:
    def __init__(self, Problem, CFL: float = 0.5):

        self.Model = Problem.model()

        self.x, self.area = Problem.geometry()
        self.dx = (self.x[1:] - self.x[:-1]).min()

        self.apply_bcs = Problem.boundary_conditions()

        self.V = Problem.initialize_primitives()
        self.U = self.Model.primitives_to_conservatives(self.V)

        self.CFL = CFL

    def solve(self, t_final):
        time = 0
        while time < t_final:
            dt = self.CFL * self.dx / self.Model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / self.dx

            F_min, F_pls = self.Model.split_flux(self.V)
            F_r = F_pls[:,1:-1] + F_min[:,2:]
            F_l = F_pls[:,:-2] + F_min[:,1:-1]

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F_r - F_l)
            self.apply_bcs(self.U)

            self.V = self.Model.conservatives_to_primitives(self.U)

        return self.Model.solution_vector(self.x, self.V)

class GodunovMethod:
    def __init__(self, Problem, CFL: float = 0.5):

        self.Model = Problem.model()

        self.x, self.area = Problem.geometry()
        self.dx = (self.x[1:] - self.x[:-1]).min()

        self.apply_bcs = Problem.boundary_conditions()

        self.V = Problem.initialize_primitives()
        self.U = self.Model.primitives_to_conservatives(self.V)
        self._nVars, self._nCells = self.V.shape

        self.CFL = CFL

    def riemann_flux(self):
        F = np.zeros((self._nVars, self._nCells-1))
        for i in range(self._nCells-1):
            left = self.Model.state_vector(self.V, i)
            right = self.Model.state_vector(self.V, i+1)

            Riemann = ExactRiemannSolver(left, right, gamma=self.Model.Thermodynamics.gamma)
            V = Riemann.sample(0)
            F[:,i] = self.Model.flux_given_state(V)
        return F

    def solve(self, t_final):
        time = 0
        while time < t_final:
            U_pred = np.zeros_like(self.U)

            dt = self.CFL * self.dx / self.Model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / self.dx

            F = self.riemann_flux()

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(self.U)

            self.V = self.Model.conservatives_to_primitives(self.U)

        return self.Model.solution_vector(self.x, self.V)


class HLLMethod:
    def __init__(self, Problem, CFL: float = 0.5):

        self.Model = Problem.model()

        self.x, self.area = Problem.geometry()
        self.dx = (self.x[1:] - self.x[:-1]).min()

        self.apply_bcs = Problem.boundary_conditions()

        self.V = Problem.initialize_primitives()
        self.U = self.Model.primitives_to_conservatives(self.V)

        self.CFL = CFL

    def solve(self, t_final):
        time = 0
        while time < t_final:
            dt = self.CFL * self.dx / self.Model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / self.dx

            F = self.Model.HLL_riemann_solver(self.V[:,:-1], self.V[:,1:])

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(self.U)

            self.V = self.Model.conservatives_to_primitives(self.U)

        return self.Model.solution_vector(self.x, self.V)

class HLLCMethod:
    def __init__(self, Problem, CFL: float = 0.5):

        self.Model = Problem.model()

        self.x, self.area = Problem.geometry()
        self.dx = (self.x[1:] - self.x[:-1]).min()

        self.apply_bcs = Problem.boundary_conditions()

        self.V = Problem.initialize_primitives()
        self.U = self.Model.primitives_to_conservatives(self.V)

        self.CFL = CFL

    def solve(self, t_final):
        time = 0
        while time < t_final:
            dt = self.CFL * self.dx / self.Model.max_characteristic(self.V)

            if time + dt > t_final:
                dt = t_final - time
            time += dt
            dt_dx = dt / self.dx

            F = self.Model.HLLC_riemann_solver(self.V[:,:-1], self.V[:,1:])

            self.U[:,1:-1] = self.U[:,1:-1] - dt_dx * (F[:,1:] - F[:,:-1])
            self.apply_bcs(self.U)

            self.V = self.Model.conservatives_to_primitives(self.U)

        return self.Model.solution_vector(self.x, self.V)