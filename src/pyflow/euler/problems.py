import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
from copy import deepcopy
import io

from ..utils.header import write_header, wrap
from .riemann import ExactRiemannSolver
from .model import CaloricallyPerfectGas, Euler

class Euler1DProblems(ABC):
    @abstractmethod
    def geometry(self):
        """Defines 1-D computational mesh."""
        pass

    @abstractmethod
    def boundary_conditions(self):
        """Returns a method defining the inflow/outflow boundary conditions for the Euler equations."""
        pass

    @abstractmethod
    def initialize_primitives(self):
        """Initialize the pressure, velocity, and density."""
        pass

    @abstractmethod
    def model(self):
        """Returns the thermodynamic state model."""
        pass


class ShocktubeProblem(Euler1DProblems):

    def __init__(self, left: Euler.State, right: Euler.State, x0: float = 0.5, Lx: float = 1.0, Nx: int = 500, gamma: float = 1.4):
        self.Lx = Lx
        self.Nx = Nx
        self.diaphragm = x0
        self.left = left
        self.right = right
        self.gamma = gamma

    def __str__(self, indent=2):
        """Return a string representation of the input parameter."""
        s = io.StringIO()
        spaces = ' '*indent
        s.write(wrap('Shocktube problem:',indent)+'\n')
        s.write(f'\n{2*spaces}Length              : {self.Lx}\n')
        s.write(f'\n{2*spaces}Diaphragm           : {self.diaphragm}\n')
        s.write(f'\n{2*spaces}Cells               : {self.Nx}\n')
        s.write(f'\n{2*spaces}Specific Heat Ratio : {self.gamma}\n')
        s.write(f'\n{2*spaces}Left State          : {self.left}\n')
        s.write(f'\n{2*spaces}Right State         : {self.right}\n')
        return s.getvalue()

    def model(self):
        return Euler(CaloricallyPerfectGas(gamma=self.gamma))

    def boundary_conditions(self):
        def supersonic_outflow(U):
            U[:,0] = U[:,1]
            U[:,-1] = U[:,-2]
        return supersonic_outflow

    def uniform_domain(self):
        return np.linspace(0, self.Lx, self.Nx)

    def geometry(self):
        return (self.uniform_domain(), np.ones((self.Nx+2,)))

    def initialize_primitives(self, ghosts=2):

        i_diaphragm = next(i for i,x in enumerate(self.uniform_domain()) if x > self.diaphragm)

        V = np.zeros((Euler.SIZE, self.Nx+ghosts))

        # Left state
        V[:,:i_diaphragm,] = Euler.State(
            density = self.left.density,
            speed = self.left.speed,
            pressure = self.left.pressure,
        ).stack()

        # Right state
        V[:,i_diaphragm:] = Euler.State(
            density = self.right.density,
            speed = self.right.speed,
            pressure = self.right.pressure,
        ).stack()

        return V

    def exact_solution(self, t_final):

        Riemann = ExactRiemannSolver(self.left, self.right, gamma=self.gamma)

        V = self.initialize_primitives(ghosts=0)

        x = self.uniform_domain()

        for i in range(len(x)):
            x_over_t = (x[i] - self.diaphragm) / t_final

            vars = Riemann.sample(x_over_t)
            V[0,i] = vars.density
            V[1,i] = vars.speed
            V[2,i] = vars.pressure

        return Euler.Solution(x=x,density=V[0],speed=V[1],pressure=V[2])