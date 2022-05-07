import io
import sys
import numpy as np
from enum import Enum
import logging
import textwrap

from .euler_equations import Euler
from .thermodynamic_model import CaloricallyPerfectGas
from .riemann import ExactRiemannSolver
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
    GODUNOV = 3
    HLL = 4
    HLLC = 5
    EXACT = 0
    NOTSET = -1
    @classmethod
    def from_name(cls, name: str):
        opts = {
            'maccormack' : cls.MACCORMACK,
            'steger-warming' : cls.STEGERWARMING,
            'godunov' : cls.GODUNOV,
            'hll' : cls.HLL,
            'hllc' : cls.HLLC,
            'exact' : cls.EXACT,
        }
        return cls(opts[name.lower()] if name.lower() in opts else cls.NOTSET)


class ShocktubeConfigError(Exception):
    """Raised when the configuration for shocktube encounters an error."""
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


class ShocktubeProblem():

    def __init__(self, left: Euler.State, right: Euler.State, method: str, x0: float = 0.5,
                 Lx: float = 1.0, Nx: int = 500, gamma: float = 1.4, CFL: float = 0.5,
                 dissipation: float = 0.0, normalize_solution=True):

        self.Lx = Lx
        self.Nx = Nx
        self.diaphragm = x0

        # Uniform computational domain
        self.x = self.uniform_mesh()

        # Internally, the computational domain is extended by one cell at each boundary
        # of the one-dimensional mesh. This cell is used for prescribing the inflow and
        # outflow boundary conditions. They are stripped before returning the solution
        # vector.
        #
        ghost_cells = 2
        self.nCells = Nx + ghost_cells

        self.left = left
        self.right = right
        self.gamma = gamma
        self.CFL = CFL
        self.normalize = normalize_solution
        self.euler = Euler(CaloricallyPerfectGas(gamma))

        self.fvm_method = Methods.from_name(method)
        if self.fvm_method == Methods.NOTSET:
            raise ShocktubeConfigError(f'{method} not a valid option.')

        elif self.fvm_method == Methods.MACCORMACK:
            log.info('Finite-volume method selected: MacCormack predictor-correct')

            self.fvm = MacCormackMethod(self.euler, dissipation=dissipation)

        elif self.fvm_method == Methods.STEGERWARMING:
            log.info('Finite-volume method selected: Steger-Warming')

            self.fvm = StegerWarmingMethod(self.euler)

        elif self.fvm_method == Methods.HLL:
            log.info('Finite-volume method selected: Flux-difference splitting, HLL')

            self.fvm = HLLMethod(self.euler)

        elif self.fvm_method == Methods.HLLC:
            log.info('Finite-volume method selected: Flux-difference splitting, HLLC')

            self.fvm = HLLCMethod(self.euler)

        else:
            log.info('Compute exact solution to shocktube problem only.')
            self.fvm = None

        self.V0 = self.initialize_primitives()

    def __str__(self, indent=2):
        """Return a string representation of the configuration parameters."""
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

    def uniform_mesh(self):
        """Create a uniform, one-dimensional, computational mesh with Nx interior
        volumes/cells.

        Returns:
            np.ndarray: Nodal coordinates of one-dimensional mesh.
        """
        nodes = self.Nx + 1
        return np.linspace(0, self.Lx, nodes)

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
            # Don't normalize the velocity
            V_normalized = self.euler.State(
                density = V[self.euler.DENSITY] / self.V0[self.euler.DENSITY,0],
                speed = V[self.euler.SPEED],
                pressure = V[self.euler.PRESSURE] / self.V0[self.euler.PRESSURE,0]
            ).stack()
            return self.euler.solution_vector(xc, V_normalized)
        else:
            return self.euler.solution_vector(xc, V)

    def solve(self, t_final:float):

        if self.fvm_method == Methods.EXACT:
            return self.exact_solution(t_final)

        # Ensure that the solution vectors are set to their initial conditions
        V = np.copy(self.V0)
        U = self.euler.primitives_to_conservatives(self.V0)

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
                U_predictor[:,1:-1] = U[:,1:-1] - dt_dx * (F_r - F_l)
                self.inflow(U_predictor)
                self.outflow(U_predictor)

                V_predictor = self.euler.conservatives_to_primitives(U_predictor)

                # Corrector
                F_l, F_r = self.fvm.corrector_flux(U_predictor, V_predictor)
                U[:,1:-1] = 0.5 * (U_predictor[:,1:-1] + U[:,1:-1] - dt_dx * (F_r - F_l))

            else:
                # Single-stage, upwind methods
                #
                F_l, F_r = self.fvm.flux(U, V)
                U[:,1:-1] = U[:,1:-1] - dt_dx * (F_r - F_l)

            # Apply the boundary conditions
            self.inflow(U)
            self.outflow(U)

            # Check that update is physical - method is not diverging
            # if self.check():
            #     break

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
        U[:,0] = U[:,1]

    def outflow(self, U):
        """Supersonic outflow boundary condition.

        Args:
            U (np.ndarray): Vector of conservative variables.
        """
        U[:,-1] = U[:,-2]

    def initialize_primitives(self, interior_only:bool = False) -> Euler.State:

        i_diaphragm = next(i for i,x in enumerate(self.x) if x > self.diaphragm)

        if interior_only:
            V = np.zeros((Euler.SIZE, self.Nx))
        else:
            V = np.zeros((Euler.SIZE, self.nCells))

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

    def exact_solution(self, t_final:float) -> Euler.Solution:

        Riemann = ExactRiemannSolver(self.left, self.right, gamma=self.gamma)

        V = self.initialize_primitives(interior_only=True)

        x = self.uniform_mesh()
        xc = 0.5 * (x[:-1] + x[1:])

        for i in range(len(xc)):
            x_over_t = (xc[i] - self.diaphragm) / t_final

            vars = Riemann.sample(x_over_t)
            V[0,i] = vars.density
            V[1,i] = vars.speed
            V[2,i] = vars.pressure

        return Euler.Solution(x=xc,density=V[0],speed=V[1],pressure=V[2])