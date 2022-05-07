import numpy as np

from .euler_equations import Euler
from .thermodynamic_model import CaloricallyPerfectGas
from .riemann import ExactRiemannSolver

class MacCormackMethod:

    def __init__(self, euler:Euler, dissipation:float = 0.0):
        self.eps = dissipation
        self._euler = euler

    def predictor_flux(self, U, V):

        F = self._euler.flux(V[:,1:])

        V_face = 0.5 * (V[:,:-1] + V[:,1:])
        F_eps = self.eps * self._euler.u_plus_c(V_face) * (U[:,1:] - U[:,:-1])
        F -= F_eps

        return (F[:,:-1], F[:,1:])

    def corrector_flux(self, U, V) -> tuple:

        F = self._euler.flux(V[:,:-1])

        V_face = 0.5 * (V[:,:-1] + V[:,1:])
        F_eps = self.eps * self._euler.u_plus_c(V_face) * (U[:,1:] - U[:,:-1])
        F -= F_eps

        return (F[:,:-1], F[:,1:])

class StegerWarmingMethod:

    def __init__(self, euler:Euler, EPS:float = 1.0e-5):
        self._euler = euler
        if not isinstance(self._euler.thermo, CaloricallyPerfectGas):
            raise RuntimeError('Steger-Warming method only implemented for CPG thermodynamics.')

        self.EPS = EPS

    def flux(self, U, V) -> tuple:

        F_min, F_pls = self.split_flux(V)
        F_r = F_pls[:,1:-1] + F_min[:,2:]
        F_l = F_pls[:,:-2] + F_min[:,1:-1]

        return (F_l, F_r)

    def split_flux(self, V):
        """Compute the split fluxes for the Steger-Warming method.

        Args:
            V (np.ndarray): Vector of primitive variables.

        Returns:
            tuple: Left-going and right-going fluxes.
        """

        c = self._euler.thermo.speed_of_sound(V[self._euler.DENSITY], V[self._euler.PRESSURE])
        c_inv = 1.0 / c

        lambdas = np.zeros_like(V)
        lambdas[0] = V[self._euler.SPEED]
        lambdas[1] = V[self._euler.SPEED] + c
        lambdas[2] = V[self._euler.SPEED] - c

        F_pls = self.compute_split_flux(
            V[self._euler.DENSITY],
            V[self._euler.SPEED],
            V[self._euler.PRESSURE],
            c_inv,
            0.5 * (lambdas + np.sqrt(lambdas**2 + self.EPS**2))
        )
        F_min = self.compute_split_flux(
            V[self._euler.DENSITY],
            V[self._euler.SPEED],
            V[self._euler.PRESSURE],
            c_inv,
            0.5 * (lambdas - np.sqrt(lambdas**2 + self.EPS**2))
        )
        return F_min, F_pls

    def compute_split_flux(self, rho, u, p, c_inv, lambdas):
        """Compute the split Euler flux-vector assuming the gas is calorically-perfect.
        For a thermally-perfect gas, the below expressions for the split fluxes would
        need to be modified.

        Args:
            rho (float): Density of the gas mixture.
            u (float): Normal velocity at the surface.
            p (float): Pressure of the gas mixture.
            c_inv (float): Inverse of the speed of sound of the gas mixture.
            lambdas (np.ndarray): Eigenvalues of the Euler equations.

        Returns:
            np.ndarray: Flux split according to the characteristics.
        """

        R1 = (rho - p * c_inv**2) * lambdas[0] + 0.5 * p * c_inv**2 * (lambdas[1] + lambdas[2])
        R2 = 0.5 * p / rho * c_inv * (lambdas[1] - lambdas[2])
        R3 = 0.5 * p * (lambdas[1] + lambdas[2])

        flux = np.zeros((3, len(u)))

        flux[0] = R1
        flux[1] = u * R1 + rho * R2
        flux[2] = 0.5 * u**2 * R1 + rho * u * R2 + R3 / (self._euler.thermo.gamma - 1)
        return flux


class GodunovMethod:

    def __init__(self, euler:Euler, EPS:float = 1.0e-5):
        self._euler = euler
        if not isinstance(self._euler.thermo, CaloricallyPerfectGas):
            raise RuntimeError('Godunov method only implemented for CPG thermodynamics.')

    def flux(self, U, V) -> tuple:

        nVars, nCells = V.shape

        F = np.zeros((nVars, nCells-1))
        for i in range(nCells-1):
            left = self._euler.state_vector(V, i)
            right = self._euler.state_vector(V, i+1)

            Riemann = ExactRiemannSolver(left, right, gamma=self._euler.thermo.gamma)
            V = Riemann.sample(0)
            F[:,i] = self._euler.model.flux_given_state(V)

        return (F[:,:-1], F[:,1:])


class HLLMethod:

    def __init__(self, euler:Euler):
        self._euler = euler
        if not isinstance(self._euler.thermo, CaloricallyPerfectGas):
            raise RuntimeError('HLL method only implemented for CPG thermodynamics.')

    def flux(self, U, V):
        F = self.HLL_riemann_solver(U[:,:-1], U[:,1:], V[:,:-1], V[:,1:])
        return (F[:,:-1], F[:,1:])

    def HLL_riemann_solver(self, U_left, U_right, V_left, V_right):
        """Compute the flux using the HLL approximate Riemann solver.

        Args:
            U_left (np.ndarray): Vector of conservatives reconstructed at face from the left.
            U_right (np.ndarray): Vector of conservatives reconstructed at face from the right.
            V_left (np.ndarray): Vector of primitives reconstructed at face from the left.
            V_right (np.ndarray): Vector of primitives reconstructed at face from the right.

        Returns:
            np.ndarray: Euler flux vector at all faces in the mesh.
        """
        F_left = self._euler.flux(V_left)
        F_right = self._euler.flux(V_right)

        cL = self._euler.thermo.speed_of_sound(V_left[self._euler.DENSITY], V_left[self._euler.PRESSURE])
        cR = self._euler.thermo.speed_of_sound(V_right[self._euler.DENSITY], V_right[self._euler.PRESSURE])

        sR = np.zeros_like(cR)
        sL = np.zeros_like(cL)

        _,Nx = V_left.shape

        for i in range(Nx):
            sL[i] = min(V_left[self._euler.SPEED,i] - cL[i], V_right[self._euler.SPEED,i] - cR[i])
            sR[i] = max(V_left[self._euler.SPEED,i] + cL[i], V_right[self._euler.SPEED,i] + cR[i])

        F = np.zeros_like(V_left)
        for i in range(Nx):
            if sL[i] >= 0:
                F[:,i] = F_left[:,i]
            elif sL[i] < 0 and sR[i] >= 0:
                F[:,i] = (sR[i] * F_left[:,i] - sL[i] * F_right[:,i] + sL[i] * sR[i] * (U_right[:,i] - U_left[:,i]))/(sR[i] - sL[i])
            else:
                F[:,i] = F_right[:,i]
        return F

class HLLCMethod:

    def __init__(self, euler:Euler):
        self._euler = euler
        if not isinstance(self._euler.thermo, CaloricallyPerfectGas):
            raise RuntimeError('HLLC method only implemented for CPG thermodynamics.')

    def flux(self, U, V):
        F = self.HLLC_riemann_solver(U[:,:-1], U[:,1:], V[:,:-1], V[:,1:])
        return (F[:,:-1], F[:,1:])

    def HLLC_riemann_solver(self, U_left, U_right, V_left, V_right):
        """Compute the flux using the HLLC approximate Riemann solver. The HLLC
        approximate Riemann solver includes the contact discontinuity.

        Args:
            U_left (np.ndarray): Vector of conservatives reconstructed at face from the left.
            U_right (np.ndarray): Vector of conservatives reconstructed at face from the right.
            V_left (np.ndarray): Vector of primitives reconstructed at face from the left.
            V_right (np.ndarray): Vector of primitives reconstructed at face from the right.

        Returns:
            np.ndarray: Euler flux vector at all faces in the mesh.
        """
        F_left = self._euler.flux(V_left)
        F_right = self._euler.flux(V_right)

        cL = self._euler.thermo.speed_of_sound(V_left[self._euler.DENSITY], V_left[self._euler.PRESSURE])
        cR = self._euler.thermo.speed_of_sound(V_right[self._euler.DENSITY], V_right[self._euler.PRESSURE])

        _,Nx = V_left.shape

        sR = np.zeros_like(cR)
        sL = np.zeros_like(cL)

        for i in range(Nx):
            sL[i] = min(V_left[self._euler.SPEED,i] - cL[i], V_right[self._euler.SPEED,i] - cR[i])
            sR[i] = max(V_left[self._euler.SPEED,i] + cL[i], V_right[self._euler.SPEED,i] + cR[i])

        rhoUL_Star = V_left[self._euler.DENSITY] * (sL - V_left[self._euler.SPEED])
        rhoUR_Star = V_right[self._euler.DENSITY] * (sR - V_right[self._euler.SPEED])
        sStar = (
            V_right[self._euler.PRESSURE] - V_left[self._euler.PRESSURE] +
            V_left[self._euler.SPEED] * rhoUL_Star - V_right[self._euler.SPEED] * rhoUR_Star) / (rhoUL_Star - rhoUR_Star)

        # def compute_star_state(rho, u, e, s, u_star):
        #     StarState = np.zeros((self.SIZE,len(rho)))
        #     StarState[0] = rho * (s - u) / (s - u_star)
        #     StarState[1] = StarState[0] * u_star
        #     StarState[2] = StarState[0] * (e / rho + (u_star - s) * (u_star + rho / (rho * (s - u))))
        #     return StarState

        # UL_Star = compute_star_state(V_left[self.DENSITY], V_left[self.SPEED], U_left[self.ENERGY], sL, sStar)
        # UR_Star = compute_star_state(V_right[self.DENSITY], V_right[self.SPEED], U_right[self.ENERGY], sR, sStar)

        UL_Star = np.zeros_like(U_left)
        UR_Star = np.zeros_like(U_right)

        UL_Star[self._euler.DENSITY] = V_left[self._euler.DENSITY] * (sL - V_left[self._euler.SPEED]) / (sL - sStar)
        UL_Star[self._euler.MOMENTUM] = UL_Star[self._euler.DENSITY] * sStar
        UL_Star[self._euler.ENERGY] = UL_Star[0] * (
            U_left[self._euler.ENERGY] / U_left[self._euler.DENSITY] + (sStar - V_left[self._euler.SPEED]) * (
                sStar + V_left[self._euler.DENSITY] / (V_left[self._euler.DENSITY] * (sL - V_left[self._euler.SPEED]))
                )
            )

        UR_Star[self._euler.DENSITY] = V_right[self._euler.DENSITY] * (sR - V_right[self._euler.SPEED]) / (sR - sStar)
        UR_Star[self._euler.MOMENTUM] = UR_Star[self._euler.DENSITY] * sStar
        UR_Star[self._euler.ENERGY] = UR_Star[0] * (
            U_right[self._euler.ENERGY] / U_right[self._euler.DENSITY] + (sStar - V_right[self._euler.SPEED]) * (
                sStar + V_left[self._euler.DENSITY] / (V_right[self._euler.DENSITY] * (sR - V_right[self._euler.SPEED]))
                )
            )

        F = np.zeros_like(V_left)
        for i in range(Nx):
            if sL[i] > 0:
                F[:,i] = F_left[:,i]

            elif sL[i] <= 0 and sStar[i] >= 0:
                F[:,i] = F_left[:,i] + sL[i] * (UL_Star[:,i] - U_left[:,i])

            elif sStar[i] <= 0 and sR[i] >= 0:
                F[:,i] = F_right[:,i] + sR[i] * (UR_Star[:,i] - U_right[:,i])

            else:
                F[:,i] = F_right[:,i]
        return F