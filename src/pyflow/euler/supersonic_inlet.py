import io
import sys
import numpy as np
from enum import Enum
import logging
import textwrap
from typing import Optional

from .euler_equations import Euler
from .thermodynamic_model import CaloricallyPerfectGas, EquationOfState
from .methods import *

logging.basicConfig(
     stream = sys.stdout,
     level = logging.INFO,
     format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

def wrap(s, indent=-1, initial_indent=-1, width=-1):
    '''Format a string of list of strings into a paragraph.'''

    if indent < 0:
        indent = 0

    if initial_indent < 0:
        initial_indent = indent

    if width < 0:
        width = 80

    if not isinstance(s,str):
        s = ' '.join(s)

    s = textwrap.fill(s.strip(),
        width=80,
        initial_indent = ' '*indent,
        subsequent_indent = ' '*indent,
        replace_whitespace = True,
    )
    s = ' '*initial_indent + s.lstrip()
    return s

class Methods(Enum):
    """Enumerator class defining the available finite-volume methods."""
    MACCORMACK = 1
    STEGERWARMING = 2
    NOTSET = -1
    @classmethod
    def from_name(cls, name: str):
        opts = {
            'maccormack' : cls.MACCORMACK,
            'steger-warming' : cls.STEGERWARMING,
        }
        return cls(opts[name.lower()] if name.lower() in opts else cls.NOTSET)


class SupersonicInletConfigError(Exception):
    """Raised when the configuration for supersonic inlet encounters an error."""
    def __init__(self, message):
        self.__help()
        super().__init__(message)

    def __help(self, indent=2):
        """Print help message when configuration error raised."""
        s = io.StringIO()
        spaces = ' '*indent
        s.write(f'\n\n')
        s.write(wrap('Available 1-D Finite-Volume Methods:',indent)+'\n')
        s.write(f'\n{2*spaces}MacCormack Predictor-Corrector   : maccormack \n')
        s.write(f'\n{2*spaces}Steger-Warming                   : steger-warming \n')
        s.write(f'\n{2*spaces}Godunov Flux-Differencing        : godunov \n')
        s.write(f'\n{2*spaces}HLL Flux-Differencing            : hll \n')
        s.write(f'\n{2*spaces}HLLC Flux-Differencing           : hllc \n')
        s.write(f'\n\n')
        return s.getvalue()


class SupersonicInletProblem():


    def __init__(self, method: str, Lx: float = 1.0, tan_theta:float = 0.25, Nx: int = 500,
                 inlet_area: float = 0.2, Rc:float = 0.5, CFL: float = 0.5, dissipation: float = 0.0,
                 p0:float = 1.0e5, T0:float = 300.0, M0:float = 2.5, gamma: float = 1.4, W:float = 28.9647,
                 pExit_p0:float = 0.0, normalize_solution=True):

        self.Lx = Lx
        self.Nx = Nx

        # Uniform computational domain
        nodes = Nx + 1
        self.x = np.linspace(0, Lx, nodes)
        self.area = self._compute_area(inlet_area, tan_theta, Rc)
        self.face_area = 0.5 * (self.area[1:] + self.area[:-1])

        # Internally, the computational domain is extended by one cell at each boundary
        # of the one-dimensional mesh. This cell is used for prescribing the inflow and
        # outflow boundary conditions. They are stripped before returning the solution
        # vector.
        #
        ghost_cells = 2
        self.nCells = Nx + ghost_cells

        self.p0 = p0
        self.T0 = T0
        self.M0 = M0
        self.gamma = gamma
        self.normalize = normalize_solution

        self.euler = Euler(CaloricallyPerfectGas(gamma))
        self._eos = EquationOfState(W)

        self.rho0 = self._eos.density(p0, T0)
        self.c0 = self.euler.thermo.speed_of_sound(self.rho0, self.p0)

        # Used for initializing a problem with a standing shock
        #
        self.pExit_p0 = pExit_p0

        # Initialize the numerical method
        #
        self.CFL = CFL
        self.fvm_method = Methods.from_name(method)
        if self.fvm_method == Methods.NOTSET:
            raise SupersonicInletConfigError(f'{method} not a valid option.')

        elif self.fvm_method == Methods.MACCORMACK:
            log.info('Finite-volume method selected: MacCormack predictor-correct')

            self.fvm = MacCormackMethod(self.euler, dissipation=dissipation)

        elif self.fvm_method == Methods.STEGERWARMING:
            log.info('Finite-volume method selected: Steger-Warming')

            self.fvm = StegerWarmingMethod(self.euler)


        self.V0 = self.initialize_primitives()
        self.U0 = self.euler.primitives_to_conservatives(self.V0)

    def __str__(self, indent=2):
        """Return a string representation of the configuration parameters."""
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
        s.write(f'\n{2*spaces}Back Pressure        : {self.p0 * self.pExit_p0}\n')

        return s.getvalue()

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
        N_mid = int(len(self.x) / 2)
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

    def _solution_state(self, V) -> Euler.Solution:
        """Pack the solution vector into a dataclass with the cell-centered mesh
        coordinates. The mesh coordinates are defined at the nodes/vertices, when
        returned as a dataclass, the cell-centered coordinates are computed and
        the ghost cells are stripped from the primitive vector.

        Args:
            V (np.ndarray): Vector of the primitive variables, density, velocity,
            and pressure at the cell-centers.

        Returns:
            Euler.Solution: A dataclass object containing the cell-centered mesh
                coordinates and the primitive variables packaged together.
        """
        # Compute the cell-centered coordinates
        xc = 0.5 * (self.x[1:] + self.x[:-1])
        if self.normalize:
            log.info('Returned solution variable speed is the Mach number')

            V_normalized = self.euler.State(
                density = V[self.euler.DENSITY] / self.rho0,
                speed = V[self.euler.SPEED] / self.euler.thermo.speed_of_sound(V[self.euler.DENSITY], V[self.euler.PRESSURE]),
                pressure = V[self.euler.PRESSURE] / self.p0
            ).stack()
            return self.euler.solution_vector(xc, V_normalized)
        else:
            return self.euler.solution_vector(xc, V)

    def solve(self, t_final:float):

        # Ensure that the solution vectors are set to their initial conditions
        V = np.copy(self.V0)
        U = self.euler.primitives_to_conservatives(self.V0)

        # Quasi-1D source term / Non-zero except for momentum in cases of area change
        S = np.zeros_like(U)

        time = 0
        iteration = 0
        while time < t_final:

            # Compute the maximum allowable timestep
            dx_min = (self.x[1:] - self.x[:-1]).min()
            dt = self.CFL * dx_min / self.euler.max_characteristic(V)
            if time + dt > t_final:
                dt = t_final - time
            time += dt

            # Common parameter used for all the below FV methods - simplified
            # because of the assumption of a uniform mesh
            dt_dx = dt / dx_min

            if self.fvm_method == Methods.MACCORMACK:

                # MacCormack's method is a two-stage, predictor-corrector method, and
                # as a result need a second level of storage, U_predictor, which is
                # then averaged with the corrector update.

                # Predictor

                F_l, F_r = self.fvm.predictor_flux(U, V)
                U_predictor = np.zeros_like(U)
                U_predictor[:,1:-1] = U[:,1:-1] - dt_dx * (F_r * self.area[1:] - F_l * self.area[:-1]) / self.face_area
                self.inflow(U_predictor)

                # Update the conservative state vector in ghost depending on whether
                # the outflow is supersonic/subsonic
                #
                U_predictor[:,-1] = self.outflow(V, dt_dx)

                V_predictor = self.euler.conservatives_to_primitives(U_predictor)

                # Corrector
                F_l, F_r = self.fvm.corrector_flux(U_predictor, V_predictor)
                U[:,1:-1] = 0.5 * (U_predictor[:,1:-1] + U[:,1:-1] - dt_dx * (F_r * self.area[1:] - F_l * self.area[:-1]) / self.face_area)

            else:
                # Single-stage, upwind methods
                #
                F_l, F_r = self.fvm.flux(U, V)
                U[:,1:-1] -= dt_dx * (F_r * self.area[1:] - F_l * self.area[:-1]) / self.face_area

            # Add the quasi-1D source term -- the pressure force from the area change driving
            # the flow through the nozzle.
            #
            S[self.euler.MOMENTUM,1:-1] = dt_dx * V[self.euler.PRESSURE,1:-1] * (self.area[1:] - self.area[:-1]) / self.face_area
            U[:,1:-1] += S[:,1:-1]

            # Apply the boundary conditions
            self.inflow(U)

            # Update the conservative state vector in ghost depending on whether
            # the outflow is supersonic/subsonic
            #
            U[:,-1] = self.outflow(V, dt_dx)

            V = self.euler.conservatives_to_primitives(U)
            iteration += 1

        print('\n')
        print('Simulation Complete')
        print(f'Final time = {time}')
        print(f'Total number of iterations = {iteration}')

        return self._solution_state(V)

    def inflow(self, U):
        """Supersonic outflow boundary condition. Boundary condition will fail
        if the rarefaction/shock wave generated within the shocktube passes
        through the inflow.

        Args:
            U (np.ndarray): Vector of conservative variables.
        """
        U[:,0] = self.U0[:,0]

    def outflow(self, V, dt_dx):
        """Characteristic outflow boundary condition handling both supersonic and subsonic outflows.

        Args:
            U (np.ndarray): Vector of conservative variables.
        """

        c = self.euler.thermo.speed_of_sound(V[self.euler.DENSITY], V[self.euler.PRESSURE])

        rho_avg = 0.5 * (V[self.euler.DENSITY,-1] + V[self.euler.DENSITY,-2])
        u_avg = 0.5 * (V[self.euler.SPEED,-1] + V[self.euler.SPEED,-2])
        p_avg = 0.5 * (V[self.euler.PRESSURE,-1] + V[self.euler.PRESSURE,-2])
        c_avg = 0.5 * (c[-1] + c[-2])

        l1 = u_avg * dt_dx
        l2 = (u_avg + c_avg) * dt_dx
        l3 = (u_avg - c_avg) * dt_dx

        c_inv = 1.0 / c_avg
        rho_c = rho_avg * c_avg

        drho = V[self.euler.DENSITY,-1] - V[self.euler.DENSITY,-2]
        du =  V[self.euler.SPEED,-1] - V[self.euler.SPEED,-2]
        dp = V[self.euler.PRESSURE,-1] - V[self.euler.PRESSURE,-2]

        dA = self.gamma * p_avg * u_avg * (self.area[-2] - self.area[-2]) / self.area[-1]

        r1 = -l1 / (1.0 + l1) * (drho - c_inv**2 * dp)
        r2 = -l2 / (1.0 + l2) * (dp + rho_c * du) - dt_dx * dA / (1.0 + l2)
        r3 = -l3 / (1.0 + l3) * (dp - rho_c * du) - dt_dx * dA / (1.0 + l3)

        if V[self.euler.SPEED,-2] / c[-2] >= 1:
            delta_p = 0.5 * (r2 + r3)
        else:
            delta_p = 0.0

        delta_rho = r1 + delta_p * c_inv**2
        delta_u = (r2 - delta_p) / rho_c

        p_bc = V[self.euler.PRESSURE,-1] + delta_p
        rho_bc = V[self.euler.DENSITY,-1] + delta_rho
        u_bc = V[self.euler.SPEED,-1] + delta_u

        Ubc = np.zeros((self.euler.SIZE,))

        Ubc[self.euler.DENSITY] = rho_bc
        Ubc[self.euler.MOMENTUM] = rho_bc * u_bc
        Ubc[self.euler.ENERGY] = p_bc / (self.gamma - 1) + 0.5 * rho_bc * u_bc**2

        return Ubc


    def initialize_primitives(self, interior_only:bool = False) -> Euler.State:
        """Initialize the flow through the supersonic inlet using isentropic gas relationships.

        Args:
            interior_only (bool, optional): Compute the flow only at interior points,
            ignoring the ghost cells. Defaults to False.

        Returns:
            Euler.State: Density, velocity, and pressure.
        """
        # Compute the area at the cell-centers
        area_avg = np.zeros((self.nCells,))
        area_avg[1:-1] = 0.5 * (self.area[1:] + self.area[:-1])
        area_avg[0] = area_avg[1]
        area_avg[-1] = area_avg[-2]

        M = self._mach_number(area_avg)
        rho, u, p = self._isentropic_flow(M, area_avg)

        if self.pExit_p0 > 0:
            p[-1] = self.pExit_p0 * p[-1]

        return Euler.State(density=rho,speed=u,pressure=p).stack()

    def _mach_number(self, area, M0:Optional[float] = None, max_error:float = 1.0e-6, max_iterations:int = 30):
        """Compute the Mach number as a function of the area change for an isentropic
        flow. This requires an iterative solution of the isentropic gas relationships.

        """

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
        """Compute the pressure, density, and velocity of an isentropic flow
        as functions of Mach number and area.

        Args:
            M (np.ndarray): Mach number
            area (np.ndarray): Area of supersonic inlet.

        Returns:
            tuple: Density, velocity, and pressure.
        """

        rho0 = self._eos.density(self.p0, self.T0)
        u0 = self.M0 * self.euler.thermo.speed_of_sound(rho0, self.p0)

        g = 0.5 * (self.gamma - 1)

        rho = rho0 * ((1 + g * self.M0**2) / (1 + g * M**2))**(1.0 / (self.gamma - 1))
        u = u0 * rho0 * area[0] / ( rho  * area )

        p = self.p0 * ((1 + g * self.M0**2) / (1 + g * M**2))**(self.gamma / (self.gamma - 1))

        return rho, u, p


