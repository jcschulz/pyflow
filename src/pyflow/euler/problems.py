from types import ClassMethodDescriptorType
import numpy as np
from dataclasses import dataclass
from typing import Union
from copy import deepcopy
import textwrap
import io

from .riemann import ExactRiemannSolver
from .model import CaloricallyPerfectGas, Euler

def write_header(s, name):
    '''Prints a section header.'''
    underline = '-'*80
    s.write(f'\n{name}\n{underline}\n')

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



class ShocktubeProblem:

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


# class ConvergingDiverginNozzle:

#     def __init__(self, case, mach, total_pressure, total_temperature, xshock: float = 0, Lx: float = 1.0, radius: float = 0.5, tan_theta: float = 0.25, Nx: int = 500, gamma: float = 1.4):

#         self.Lx = Lx
#         self.tan_theta = tan_theta
#         self.Rc = radius
#         self.Nx = Nx
#         self.gamma = gamma
#         self.xshock = xshock

#         self.case = case
#         self.mach = mach
#         self.pressure = total_pressure
#         self.temperature= total_temperature

#     def model(self):
#         return Euler(CaloricallyPerfectGas(gamma=self.gamma))

#     def uniform_domain(self):
#         return np.linspace(0, self.Lx, self.Nx)

#     def geometry(self):
#         """
#         Define the geometry of the converging-diverging nozzle.

#         Args:
#             x (np.array) : One-dimensional array of x-coordinates along nozzle.
#             tan_theta (float) : Tangent of the angle defining the slope of the converging
#                 and diverging sections of the nozzle.
#             Rc (float) : Radius of curvature rounding the intersection of the tangent lines
#                 defining the converging and diverging slopes of the nozzle.

#         Returns:
#             y (np.array) : One-dimensional array of y-coordinates along nozzle. The y-distance
#                 is also the cross-section area normalized by the in-plane length.

#         """
#         # Compute the total length of inlet with the number of mesh points. Assume the
#         # mesh is uniformally spaced.
#         x = self.uniform_domain()
#         dx = x[1] - x[0]

#         # The inlet is a symmetric converging-diverging nozzle, so we will compute the
#         # y-coordinates for the converging section and the reflect across the throat.
#         N_mid = int(self.Nx/2)
#         L_mid = 0.5 * self.Lx
#         x_half = np.linspace(0,L_mid,N_mid)

#         # Angle of the converging/diverging section of the inlet.
#         theta = np.arctan(self.tan_theta)

#         # Find the intersection of the rounded mid-section with the tangent line. At
#         # (x1,y1) switch from linear slope to a slope defined by the radius of curvature.
#         r1 = L_mid / np.cos(theta)
#         r2 = L_mid * self.tan_theta

#         x1 = (r1 - r2) * np.cos(theta)
#         y1 =  x1 * self.tan_theta

#         # The radius of curvature is defined by a circle with the origin at x = L_mid and
#         # y = yc. Compute the y-origin given the known intersection point (x1,y1).
#         yc = y1 - np.sqrt(self.Rc**2 - (x1 - L_mid)**2)

#         # Get the y mesh points.
#         y1_mesh = x_half * self.tan_theta
#         y2_mesh = yc + np.sqrt(self.Rc**2 - (x_half - L_mid)**2)

#         y_half = np.where(x_half > x1, y2_mesh, y1_mesh)

#         # Mirror the mesh across the throat.
#         y = np.zeros_like(x)
#         y[:N_mid] = y_half
#         y[N_mid] = y_half[-1]
#         y[N_mid+1:] = np.flip(y_half)

#         return x, y

#     def initialize_primitives(self, ghosts=2):

#         V = np.zeros((Euler.SIZE, self.Nx+ghosts))

#         x, area = self.geometry()
#         gas_constant =  8314.46261815324 / 28.9647

#         if self.case == 'uniform':
#             rho0 = self.pressure / gas_constant / self.temperature
#             c0 = np.sqrt(self.gamma * gas_constant * self.temperature)
#             u0 = c0 * self.mach

#             M = self.mach_number(self.mach, self.area, self.gamma)
#             rho, u, p = self.isentropic_flow(rho0, u0, self.pressure, M, self.area, self.gamma)

#         else:
#             rho0 = self.pressure / gas_constant / self.temperature
#             c0 = np.sqrt(self.gamma * gas_constant * self.temperature)
#             u0 = c0 * self.mach


#             # Flow before the standing-shock
#             M = self.mach_number(self.mach, self.area, self.gamma)
#             rho, u, p = self.isentropic_flow(rho0, u0, self.pressure, M, self.area, self.gamma)

#             # Jump conditions
#             ishock = int(self.xshock / (x[1] - x[0]))

#             mass = rho[ishock-1] * u[ishock-1]
#             momentum = rho[ishock-1] * u[ishock-1]**2 + p[ishock-1]
#             e = p[ishock-1] / (self.gamma - 1) / rho[ishock-1] + 0.5 * u[ishock-1]**2
#             h = e + p[ishock-1] / rho[ishock-1]

#             a = 0.5 - self.gamma / (self.gamma - 1)
#             b = self.gamma / (self.gamma - 1) * momentum / mass
#             c = -h

#             u1_pls = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
#             u1_min = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
#             if u1_pls < u1_min:
#                 u1 = u1_pls
#             else:
#                 u1 = u1_min
#             rho1 = mass / u1
#             p1 = momentum - rho1 * u1**2
#             c1 = np.sqrt(self.gamma * p1 / rho1)
#             M1 = u1 / c1

#             area1 = area[ishock:]

#             M = self.mach_number(M1, area[ishock:], self.gamma)
#             rho[ishock:], u[ishock:], p[ishock:] = self.isentropic_flow(rho1, u1, p1, M, area1, self.gamma)

#         V = Euler.State(density=rho, speed=u, pressure=p)
#         return V

#     @staticmethod
#     def mach_number(M0, area, gamma):
#         MaxError = 1.0e-6
#         MaxIterations = 30

#         A_ratio = area / area[0]
#         M = M0 * np.ones_like(A_ratio)

#         g = 0.5 * (gamma - 1)
#         g_exp = (gamma + 1) / (gamma - 1)

#         error = 1.0
#         i = 0
#         while error > MaxError and i < MaxIterations:
#             f = A_ratio**2 - M0**2 / M**2 * ( (1 + g * M**2) / (1 + g * M0**2) )**g_exp

#             dfdM = ( 2 * M0**2 / M**3 * ( (1 + g * M**2) / (1 + g * M0**2) )**g_exp -
#                     (gamma + 1) / M * M0**2 * (1 / (1 + g * M0**2))**g_exp * (1 + g * M**2)**(g_exp - 1)
#                 )

#             M -= f / dfdM

#             i += 1
#             error = np.max(np.abs(f))

#         return M

#     @staticmethod
#     def isentropic_flow(rho0, u0, p0, M, area, gamma):

#         M0 = u0 / np.sqrt(gamma * p0 / rho0)
#         g = 0.5 * (gamma - 1)

#         rho = rho0 * ((1 + g * M0**2) / (1 + g * M**2))**(1.0 / (gamma - 1))

#         u = u0 * rho0 * area[0] / ( rho  * area )

#         p = p0 * ((1 + g * M0**2) / (1 + g * M**2))**(gamma / (gamma - 1))

#         return rho, u, p


#     def boundary_conditions(self):
#         def supersonic_inflow_characteristic_outflow(dt_dx, rho, u, p, c, area, gamma):

#             rho_avg = 0.5 * (rho[-1] + rho[-2])
#             u_avg = 0.5 * (u[-1] + u[-2])
#             c_avg = 0.5 * (c[-1] + c[-2])
#             p_avg = 0.5 * (p[-1] + p[-2])

#             l1 = u_avg * dt_dx
#             l2 = (u_avg + c_avg) * dt_dx
#             l3 = (u_avg - c_avg) * dt_dx

#             c_inv = 1.0 / c[-1]
#             rho_c = rho_avg * c_avg

#             drho = rho[-1] - rho[-2]
#             du = u[-1] - u[-2]
#             dp = p[-1] - p[-2]

#         #     dA = gamma * p[-1] * u[-1] * (area[-1] - area[-2]) / area[-1]
#             dA = gamma * p_avg * u_avg * (area[-1] - area[-2]) / area[-1]

#             r1 = -l1 / (1 + l1) * (drho - c_inv**2 * dp)
#             r2 = -(l2 * (dp + rho_c * du) - dt_dx * dA) / (1 + l2)
#             r3 = -(l3 * (dp - rho_c * du) - dt_dx * dA) / (1 + l3)

#             if u[-2] / c[-2] >= 1:
#                 delta_p = 0.5 * (r2 + r3)
#             else:
#                 delta_p = 0.0

#             delta_rho = r1 + delta_p * c_inv**2
#             delta_u = (r2 - delta_p) / rho_c

#             p_bc = p[-1] + delta_p
#             rho_bc = rho[-1] + delta_rho
#             u_bc = u[-1] + delta_u

#             dQ = np.zeros((3))

#             dQ[0] = rho_bc - rho[-1]
#             dQ[1] = rho_bc * u_bc - rho[-1] * u[-1]
#             e_int_bc = p_bc / (gamma - 1)
#             e_int = p[-1] / (gamma - 1)
#             dQ[2] = e_int_bc + 0.5 * rho_bc * u_bc**2 - e_int - 0.5 * rho[-1] * u[-1]**2

#             return dQ, rho_bc, u_bc, p_bc

#         return supersonic_inflow_characteristic_outflow










