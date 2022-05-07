import numpy as np
from dataclasses import dataclass
from typing import Union
import sys
import logging
from copy import deepcopy

from .thermodynamic_model import ThermodynamicModel

logging.basicConfig(
     stream = sys.stdout,
     level = logging.INFO,
     format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class PrimitiveState:
    """Primitive state vector of the one-dimensional Euler equations.

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
    """Object containing data and methods used for solving the Euler equations.

    Args:
        thermodynamic_model (ThermodynamicModel): _description_
    """

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

    def __init__(self, thermodynamic_model:ThermodynamicModel):
        self.thermo = thermodynamic_model

    def flux(self, V_: np.ndarray) -> np.ndarray:
        """Comptue the Euler fluxes.

        Args:
            V_ (np.ndarray): Vector of primitive variables.

        Returns:
            np.ndarray: Euler flux vector.
        """
        F = np.zeros_like(V_)
        V = deepcopy(V_)
        F[self.DENSITY] = V[self.DENSITY] * V[self.SPEED]
        F[self.MOMENTUM] = V[self.DENSITY] * V[self.SPEED]**2 + V[self.PRESSURE]

        energy = self.thermo.totalenergy(V[self.DENSITY], V[self.PRESSURE], V[self.SPEED])

        F[self.ENERGY] = V[self.SPEED] * (energy  + V[self.PRESSURE])
        return F

    def solution_vector(self, x:np.ndarray, V:np.ndarray) -> Solution:
        """Pack the primitive variables with the cell-centered coordinates into
        a plain dataclass. The ghost cells are stripped from the solution.

        Args:
            x (np.ndarray): Coordinates of the mesh.
            V (np.ndarray): Vector of primitive variables.

        Returns:
            Solution: Plain dataclass containing the solution state.
        """
        return self.Solution(*(v.flatten() for v in np.split(V[:,1:-1],self.SIZE)),x)

    def state_vector(self, V:np.ndarray, index:int) -> PrimitiveState:
        """Pack the primitive variable vector into a plain dataclass at
        a specified index within the computational mesh.

        Args:
            V (np.ndarray): Vector of primitive variables
            index (int): Index within the computational mesh.

        Returns:
            PrimitiveState: Vector of primitive variables.
        """
        return self.State(
            density=V[self.DENSITY,index],
            speed=V[self.SPEED,index],
            pressure=V[self.PRESSURE,index]
        )

    def flux_given_state(self, V:PrimitiveState) -> np.ndarray:
        """Compute the Euler fluxes given the a dataclass of the primitive variables.

        Args:
            V (PrimitiveState): Dataclass containing the primitive variables.

        Returns:
            np.ndarray: Vector of the Euler fluxes.
        """
        F = np.zeros((self.SIZE,))
        F[self.DENSITY] = V.density * V.speed
        F[self.MOMENTUM] = V.density * V.speed**2 + V.pressure

        energy = self.thermo.internalenergy(V.density, V.pressure) + 0.5 * V.speed**2
        F[self.ENERGY] = V.speed * (V.density * energy  + V.pressure)
        return F

    def u_plus_c(self, V:np.ndarray) -> np.ndarray:
        """Compute the right-going characteristic velocity.

        Args:
            V (np.ndarray): Vector of primitive variables.

        Returns:
            float: Right-going characteristic.
        """
        return V[self.SPEED] + self.thermo.speed_of_sound(V[self.DENSITY], V[self.PRESSURE])

    def max_characteristic(self, V:np.ndarray) -> float:
        """Compute the maximum characteristic velocity within the computational domain.

        Args:
            V (np.ndarray): Vector of primitive variables.

        Returns:
            float: Maximum characteristic.
        """
        return np.max(self.u_plus_c(V))

    def primitives_to_conservatives(self, V_:np.ndarray) -> np.ndarray:
        """Compute the conservative variables from the primitive variables.

        Args:
            V_ (np.ndarray): Vector of primitive variables.

        Returns:
            np.ndarray: Vector of conservative variables.
        """
        U = np.zeros_like(V_)
        V = deepcopy(V_)
        U[self.DENSITY] = V[self.DENSITY]
        U[self.MOMENTUM] = V[self.DENSITY] * V[self.SPEED]
        U[self.ENERGY] = self.thermo.totalenergy(V[self.DENSITY], V[self.PRESSURE], V[self.SPEED])
        return U

    def conservatives_to_primitives(self, U_:np.ndarray) -> np.ndarray:
        """Compute the primitive variables from the conservative variables.

        Args:
            V_ (np.ndarray): Vector of conservative variables.

        Returns:
            np.ndarray: Vector of primitive variables.
        """
        V = np.zeros_like(U_)
        U = deepcopy(U_)
        V[self.DENSITY] = U[self.DENSITY]
        V[self.SPEED] = U[self.MOMENTUM] / U[self.DENSITY]
        V[self.PRESSURE] = self.thermo.pressure(U[self.DENSITY], V[self.SPEED], U[self.ENERGY])
        return V