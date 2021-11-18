import numpy as np
from collections import namedtuple

PrimitiveState = namedtuple('PrimitiveState', ['rho', 'u', 'p'])
"""For the one-dimensional Euler equations, the primitive state vector for
a gas mixture is defined by the density, velocity (speed), and pressure.

Attributes:
    rho (float): Density of the gas mixture [kg/m^3].
    u (float): Velocity of the gas mixture [m/s].
    p (float): Pressure of the gas mixture [Pa].
"""

ConservedState = namedtuple('ConservedState', ['rho', 'rhoU', 'E'])
"""For the one-dimensional Euler equations, the conserved state vector for
a gas mixture is defined by conserved quantities of mass (density), momentum,
and total energy.

Attributes:
    rho (float): Density in units of kg/m^3.
    rhoU (float): Momentum in units of kg/(m^2-s).
    E (float): Total energy in units of J/m^3.
"""
