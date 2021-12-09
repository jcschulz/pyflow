from types import prepare_class
import numpy as np
from dataclasses import dataclass, InitVar
from typing import Union
from enum import Enum
import logging
from copy import deepcopy


@dataclass
class PrimitiveState:
    """Primitive state vector of the one-dimensional Euler equations for
    a calorically-perfect gas mixture.

    Args:
        density (Union[float,np.ndarray]): Density of the gas mixture [kg/m^3].
        speed (Union[float,np.ndarray]): Velocity of the gas mixture [m/s].
        pressure (Union[float,np.ndarray]): Pressure of the gas mixture [Pa].
    """
    density: Union[float,np.ndarray]
    speed: Union[float,np.ndarray]
    pressure: Union[float,np.ndarray]

    @property
    def size(self):
        return 3

    def stack(self) -> np.ndarray:
        """Return a primitive state vector as a stack of numpy arrays.

        Returns:
            np.ndarray: A stack of density, speed, and pressure states.
        """
        return np.stack((np.atleast_1d(self.density), np.atleast_1d(self.speed), np.atleast_1d(self.pressure)))


class Euler:

    State = PrimitiveState

    DENSITY = 0
    MOMENTUM = 1
    SPEED = 1
    ENERGY = 2
    PRESSURE = 2
    SIZE = 3

    @dataclass
    class Solution(State):
        x : np.ndarray

    def __init__(self, ThermodynamicModel):
        self.Thermodynamics = ThermodynamicModel

    def flux(self, V_: np.ndarray) -> np.ndarray:
        F = np.zeros_like(V_)
        V = deepcopy(V_)
        F[self.DENSITY] = V[self.DENSITY] * V[self.SPEED]
        F[self.MOMENTUM] = V[self.DENSITY] * V[self.SPEED]**2 + V[self.PRESSURE]

        energy = self.Thermodynamics.totalenergy(V[self.DENSITY], V[self.PRESSURE], V[self.SPEED])

        F[self.ENERGY] = V[self.SPEED] * (energy  + V[self.PRESSURE])
        return F

    def solution_vector(self, x, V):
        return self.Solution(*(v.flatten() for v in np.split(V[:,1:-1],self.SIZE)),x)

    def state_vector(self, V, Index):
        return self.State(
            density=V[self.DENSITY,Index],
            speed=V[self.SPEED,Index],
            pressure=V[self.PRESSURE,Index]
        )

    def flux_given_state(self, V: PrimitiveState) -> np.ndarray:
        F = np.zeros((self.SIZE,))
        F[self.DENSITY] = V.density * V.speed
        F[self.MOMENTUM] = V.density * V.speed**2 + V.pressure

        energy = self.Thermodynamics.internalenergy(V.density, V.pressure) + 0.5 * V.speed**2
        F[self.ENERGY] = V.speed * (V.density * energy  + V.pressure)
        return F

    def u_plus_c(self, V: np.ndarray) -> np.ndarray:
        return V[self.SPEED] + self.Thermodynamics.speed_of_sound(V[self.DENSITY], V[self.PRESSURE])

    def max_characteristic(self, V: np.ndarray) -> float:
        return np.max(V[self.SPEED] + self.Thermodynamics.speed_of_sound(V[self.DENSITY], V[self.PRESSURE]))

    def primitives_to_conservatives(self, V_: np.ndarray) -> np.ndarray:
        U = np.zeros_like(V_)
        V = deepcopy(V_)
        U[self.DENSITY] = V[self.DENSITY]
        U[self.MOMENTUM] = V[self.DENSITY] * V[self.SPEED]
        U[self.ENERGY] = self.Thermodynamics.totalenergy(V[self.DENSITY], V[self.PRESSURE], V[self.SPEED])
        return U

    def conservatives_to_primitives(self, U_: np.ndarray) -> np.ndarray:
        V = np.zeros_like(U_)
        U = deepcopy(U_)
        V[self.DENSITY] = U[self.DENSITY]
        V[self.SPEED] = U[self.MOMENTUM] / U[self.DENSITY]
        V[self.PRESSURE] = self.Thermodynamics.pressure(U[self.DENSITY], V[self.SPEED], U[self.ENERGY])
        return V

    def HLL_riemann_solver(self, V_left: np.ndarray, V_right: np.ndarray) -> np.ndarray:
        F_left = self.flux(V_left)
        F_right = self.flux(V_right)

        U_left = self.primitives_to_conservatives(V_left)
        U_right = self.primitives_to_conservatives(V_right)

        cL = self.Thermodynamics.speed_of_sound(V_left[self.DENSITY], V_left[self.PRESSURE])
        cR = self.Thermodynamics.speed_of_sound(V_right[self.DENSITY], V_right[self.PRESSURE])

        sR = np.zeros_like(cR)
        sL = np.zeros_like(cL)

        _,Nx = V_left.shape

        for i in range(Nx):
            sL[i] = min(V_left[self.SPEED,i] - cL[i], V_right[self.SPEED,i] - cR[i])
            sR[i] = max(V_left[self.SPEED,i] + cL[i], V_right[self.SPEED,i] + cR[i])

        F = np.zeros_like(V_left)
        for i in range(Nx):
            if sL[i] >= 0:
                F[:,i] = F_left[:,i]
            elif sL[i] < 0 and sR[i] >= 0:
                F[:,i] = (sR[i] * F_left[:,i] - sL[i] * F_right[:,i] + sL[i] * sR[i] * (U_right[:,i] - U_left[:,i]))/(sR[i] - sL[i])
            else:
                F[:,i] = F_right[:,i]
        return F

    def HLLC_riemann_solver(self, V_left: np.ndarray, V_right: np.ndarray) -> np.ndarray:
        F_left = self.flux(V_left)
        F_right = self.flux(V_right)

        U_left = self.primitives_to_conservatives(V_left)
        U_right = self.primitives_to_conservatives(V_right)

        cL = self.Thermodynamics.speed_of_sound(V_left[self.DENSITY], V_left[self.PRESSURE])
        cR = self.Thermodynamics.speed_of_sound(V_right[self.DENSITY], V_right[self.PRESSURE])

        _,Nx = V_left.shape

        sR = np.zeros_like(cR)
        sL = np.zeros_like(cL)

        for i in range(Nx):
            sL[i] = min(V_left[self.SPEED,i] - cL[i], V_right[self.SPEED,i] - cR[i])
            sR[i] = max(V_left[self.SPEED,i] + cL[i], V_right[self.SPEED,i] + cR[i])

        rhoUL_Star = V_left[self.DENSITY] * (sL - V_left[self.SPEED])
        rhoUR_Star = V_right[self.DENSITY] * (sR - V_right[self.SPEED])
        sStar = (
            V_right[self.PRESSURE] - V_left[self.PRESSURE] +
            V_left[self.SPEED] * rhoUL_Star - V_right[self.SPEED] * rhoUR_Star) / (rhoUL_Star - rhoUR_Star)

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

        UL_Star[self.DENSITY] = V_left[self.DENSITY] * (sL - V_left[self.SPEED]) / (sL - sStar)
        UL_Star[self.MOMENTUM] = UL_Star[self.DENSITY] * sStar
        UL_Star[self.ENERGY] = UL_Star[0] * (
            U_left[self.ENERGY] / U_left[self.DENSITY] + (sStar - V_left[self.SPEED]) * (
                sStar + V_left[self.DENSITY] / (V_left[self.DENSITY] * (sL - V_left[self.SPEED]))
                )
            )

        UR_Star[self.DENSITY] = V_right[self.DENSITY] * (sR - V_right[self.SPEED]) / (sR - sStar)
        UR_Star[self.MOMENTUM] = UR_Star[self.DENSITY] * sStar
        UR_Star[self.ENERGY] = UR_Star[0] * (
            U_right[self.ENERGY] / U_right[self.DENSITY] + (sStar - V_right[self.SPEED]) * (
                sStar + V_left[self.DENSITY] / (V_right[self.DENSITY] * (sR - V_right[self.SPEED]))
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

    def split_flux(self, V, EPS: float = 1.0e-3):
        '''
        Compute the split fluxes for the Steger-Warming method.
        '''

        c = self.Thermodynamics.speed_of_sound(V[self.DENSITY], V[self.PRESSURE])
        c_inv = 1.0 / c

        lambdas = np.zeros_like(V)
        lambdas[0] = V[self.SPEED]
        lambdas[1] = V[self.SPEED] + c
        lambdas[2] = V[self.SPEED] - c

        lambda_pls = 0.5 * (lambdas + np.sqrt(lambdas**2 + EPS**2))
        lambda_min = 0.5 * (lambdas - np.sqrt(lambdas**2 + EPS**2))

        F_pls = self.compute_split_flux(V[self.DENSITY], V[self.SPEED], V[self.PRESSURE], c_inv, lambda_pls)
        F_min = self.compute_split_flux(V[self.DENSITY], V[self.SPEED], V[self.PRESSURE], c_inv, lambda_min)
        return F_min, F_pls

    def compute_split_flux(self, rho, u, p, c_inv, lambdas):

        R1 = (rho - p * c_inv**2) * lambdas[0] + 0.5 * p * c_inv**2 * (lambdas[1] + lambdas[2])
        R2 = 0.5 * p / rho * c_inv * (lambdas[1] - lambdas[2])
        R3 = 0.5 * p * (lambdas[1] + lambdas[2])

        flux = np.zeros((3, len(u)))

        flux[0] = R1
        flux[1] = u * R1 + rho * R2
        flux[2] = 0.5 * u**2 * R1 + rho * u * R2 + R3 / (self.Thermodynamics.gamma - 1)
        return flux


class CaloricallyPerfectGas:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def speed_of_sound(self, density, pressure):
        return np.sqrt(self.gamma * pressure / density)

    def internalenergy(self, density, pressure):
        return pressure / density / (self.gamma - 1.0)

    def totalenergy(self, density, pressure, u):
        return pressure / (self.gamma - 1) + 0.5 * density * u**2

    def pressure(self, density, u, total_energy):
        return (total_energy - 0.5 * density * u**2 ) * (self.gamma - 1)
