import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
import io

from ..utils.header import write_header, wrap
from .model import CaloricallyPerfectGas, Euler


class EulerQuasi1DProblem(ABC):
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

class SupersonicInlet(EulerQuasi1DProblem):

    def __init__(self, Lx: float = 1.0, tan_theta:float = 0.25, inlet_area: float = 0.2, Rc:float = 0.5,
            Nx: int = 500, p0:float = 1.0e5, T0:float = 300.0, M0:float = 2.5, gamma: float = 1.4,
            W:float = 28.9647, pExit_p0:float = 0.0):

        self.Lx = Lx
        self.Nx = Nx - 1
        self.p0 = p0
        self.T0 = T0
        self.M0 = M0
        self.gamma = gamma
        self._cpg = CaloricallyPerfectGas(gamma=gamma, W=W)
        self.rho0 = self._cpg.density(self.p0, self.T0)
        self.c0 = self._cpg.speed_of_sound(self.rho0, self.p0)
        self.x = np.linspace(0, Lx, Nx)
        self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        self.area = self._compute_area(inlet_area, tan_theta, Rc)

        # Used for initializing a problem with a standing shock
        #
        self.pExit_p0 = pExit_p0

    def _compute_area(self, inlet_area:float, tan_theta:float, Rc:float) -> np.ndarray:
        """
        Define the geometry of the converging-diverging nozzle.
        Assume the mesh is uniformally spaced.

        Args:
            inlet_area (float) : Area of the inlet.
            tan_theta (float) : Tangent of the angle defining the slope of the converging
                and diverging sections of the nozzle.
            Rc (float) : Radius of curvature rounding the intersection of the tangent lines
                defining the converging and diverging slopes of the nozzle.

        Returns:
            y (np.array) : One-dimensional array of y-coordinates along nozzle. The y-distance
                is also the cross-section area normalized by the in-plane length.

        """

        # The inlet is a symmetric converging-diverging nozzle, so we will compute the
        # y-coordinates for the converging section and the reflect across the throat.
        N_mid = int(self.Nx / 2)
        L_mid = 0.5 * self.Lx
        x_half = np.linspace(0,L_mid,N_mid)

        # Angle of the converging/diverging section of the inlet.
        theta = np.arctan(tan_theta)

        # Find the intersection of the rounded mid-section with the tangent line. At
        # (x1,y1) switch from linear slope to a slope defined by the radius of curvature.
        r1 = L_mid / np.cos(theta)
        r2 = L_mid * tan_theta

        x1 = (r1 - r2) * np.cos(theta)
        y1 =  x1 * tan_theta

        # The radius of curvature is defined by a circle with the origin at x = L_mid and
        # y = yc. Compute the y-origin given the known intersection point (x1,y1).
        yc = y1 - np.sqrt(Rc**2 - (x1 - L_mid)**2)

        # Get the y mesh points.
        y1_mesh = x_half * tan_theta
        y2_mesh = yc + np.sqrt(Rc**2 - (x_half - L_mid)**2)

        y_half = np.where(x_half > x1, y2_mesh, y1_mesh)

        # Mirror the mesh across the throat.
        y = np.zeros_like(self.x)
        y[:N_mid] = y_half
        y[N_mid] = y_half[-1]
        y[N_mid+1:] = np.flip(y_half)
        return y - inlet_area

    def __str__(self, indent=2):
        """Return a string representation of the input parameter."""
        s = io.StringIO()
        spaces = ' '*indent
        s.write(wrap('Supersonic Inlet problem:',indent)+'\n')
        s.write(f'\n{2*spaces}Length               : {self.Lx}\n')
        s.write(f'\n{2*spaces}Throat Area          : {np.min(np.abs(self.area))}\n')
        s.write(f'\n{2*spaces}Cells                : {self.Nx}\n')
        s.write(f'\n{2*spaces}Specific Heat Ratio  : {self.gamma}\n')
        s.write(f'\n{2*spaces}Inlet Mach Number    : {self.M0}\n')
        s.write(f'\n{2*spaces}Inlet Pressure       : {self.p0}\n')
        s.write(f'\n{2*spaces}Inlet Density        : {self.rho0}\n')
        s.write(f'\n{2*spaces}Inlet Speed of Sound : {self.c0}\n')

        return s.getvalue()

    def model(self):
        return Euler(CaloricallyPerfectGas(gamma=self.gamma))

    def boundary_conditions(self):
        def outflow_bc(dt_dx, V):

            c = self._cpg.speed_of_sound(V[Euler.DENSITY], V[Euler.PRESSURE])

            rho_avg = 0.5 * (V[Euler.DENSITY,-1] + V[Euler.DENSITY,-2])
            u_avg = 0.5 * (V[Euler.SPEED,-1] + V[Euler.SPEED,-2])
            p_avg = 0.5 * (V[Euler.PRESSURE,-1] + V[Euler.PRESSURE,-2])
            c_avg = 0.5 * (c[-1] + c[-2])

            l1 = u_avg * dt_dx
            l2 = (u_avg + c_avg) * dt_dx
            l3 = (u_avg - c_avg) * dt_dx

            c_inv = 1.0 / c_avg
            rho_c = rho_avg * c_avg

            drho = V[Euler.DENSITY,-1] - V[Euler.DENSITY,-2]
            du =  V[Euler.SPEED,-1] - V[Euler.SPEED,-2]
            dp = V[Euler.PRESSURE,-1] - V[Euler.PRESSURE,-2]

            dA = self.gamma * p_avg * u_avg * (self.area[-1] - self.area[-2]) / self.area[-1]

            r1 = -l1 / (1.0 + l1) * (drho - c_inv**2 * dp)
            r2 = -l2 / (1.0 + l2) * (dp + rho_c * du) - dt_dx * dA / (1.0 + l2)
            r3 = -l3 / (1.0 + l3) * (dp - rho_c * du) - dt_dx * dA / (1.0 + l3)

            if V[Euler.SPEED,-2] / c[-2] >= 1:
                delta_p = 0.5 * (r2 + r3)
            else:
                delta_p = 0.0

            delta_rho = r1 + delta_p * c_inv**2
            delta_u = (r2 - delta_p) / rho_c

            p_bc = V[Euler.PRESSURE,-1] + delta_p
            rho_bc = V[Euler.DENSITY,-1] + delta_rho
            u_bc = V[Euler.SPEED,-1] + delta_u

            dU = np.zeros((Euler.SIZE))

            dU[Euler.DENSITY] = rho_bc
            dU[Euler.MOMENTUM] = rho_bc * u_bc
            dU[Euler.ENERGY] = p_bc / (self.gamma - 1) + 0.5 * rho_bc * u_bc**2
            return dU

        return outflow_bc

    def geometry(self):
        return (self.x, self.xc, self.area)

    def initialize_primitives(self, ghosts=2):

        # Compute the area at the cell-centers
        area_avg = np.zeros((self.Nx + ghosts,))
        area_avg[1:-1] = 0.5 * (self.area[1:] + self.area[:-1])
        area_avg[0] = area_avg[1]
        area_avg[-1] = area_avg[-2]

        M = self._mach_number(area_avg)
        rho, u, p = self._isentropic_flow(M, area_avg)

        if self.pExit_p0 > 0:
            p[-1] = self.pExit_p0 * p[-1]

        return Euler.State(density=rho,speed=u,pressure=p).stack()

    def _mach_number(self, area, M0:Optional[float] = None, max_error:float = 1.0e-6, max_iterations:int = 30):

        if M0 is not None:
            Ma = M0
        else:
            Ma = self.M0

        A_ratio = area / area[0]
        M = Ma * np.ones_like(A_ratio)

        g = 0.5 * (self.gamma - 1)
        g_exp = (self.gamma + 1) / (self.gamma - 1)

        error = 1.0
        iteration = 0
        while error > max_error and iteration < max_iterations:
            f = A_ratio**2 - Ma**2 / M**2 * ( (1 + g * M**2) / (1 + g * Ma**2) )**g_exp

            dfdM = ( 2 * Ma**2 / M**3 * ( (1 + g * M**2) / (1 + g * Ma**2) )**g_exp -
                    (self.gamma + 1) / M * Ma**2 * (1 / (1 + g * Ma**2))**g_exp * (1 + g * M**2)**(g_exp - 1)
                )

            M -= f / dfdM

            iteration += 1
            error = np.max(np.abs(f))
        return M

    def _isentropic_flow(self, M, area):

        rho0 = self._cpg.density(self.p0, self.T0)
        u0 = self.M0 * self._cpg.speed_of_sound(rho0, self.p0)

        g = 0.5 * (self.gamma - 1)

        rho = rho0 * ((1 + g * self.M0**2) / (1 + g * M**2))**(1.0 / (self.gamma - 1))
        u = u0 * rho0 * area[0] / ( rho  * area )

        p = self.p0 * ((1 + g * self.M0**2) / (1 + g * M**2))**(self.gamma / (self.gamma - 1))

        return rho, u, p

