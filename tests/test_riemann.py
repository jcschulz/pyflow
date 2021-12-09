import pytest
import numpy as np

from pyflow.euler.model import PrimitiveState, CaloricallyPerfectGas
from pyflow.euler.riemann import ExactRiemannSolver, StarState

@pytest.fixture
def RP1():
    return {
        'left' : PrimitiveState(density=1.0, speed=0.0, pressure=1.0),
        'right': PrimitiveState(density=0.125, speed=0.0, pressure=0.1),
        'p_star' : 0.30313,
        'u_star' : 0.92745,
        'rhoL_star' : 0.42632,
        'rhoR_star' : 0.26557
    }

def test_starstate_RP1(RP1):
    """Test computation of Star state for Riemann Problem 1"""
    CPG = CaloricallyPerfectGas(gamma=1.4)
    p_star, u_star = StarState(RP1['left'], RP1['right'], CPG).find()

    assert p_star.round(5) == RP1['p_star']
    assert u_star.round(5) == RP1['u_star']

def test_RP1(RP1):
    """Test Riemann Problem 1"""
    Riemann = ExactRiemannSolver(RP1['left'], RP1['right'])

    V = Riemann.sample(-2.5)
    assert V.density == 1.0
    assert V.speed == 0.0
    assert V.pressure == 1.0

    # Point inside rarefaction wave: See Sod paper
    V = Riemann.sample(-0.9869739478957917)
    assert V.density.round(3) == 0.869
    assert V.speed.round(3) == 0.164
    assert V.pressure.round(3) == 0.822

    V = Riemann.sample(0.0)
    assert V.density.round(5) == RP1['rhoL_star']
    assert V.speed.round(5) == RP1['u_star']
    assert V.pressure.round(5) == RP1['p_star']

    V = Riemann.sample(1.0)
    assert V.density.round(5) == RP1['rhoR_star']
    assert V.speed.round(5) == RP1['u_star']
    assert V.pressure.round(5) == RP1['p_star']

@pytest.fixture
def RP2():
    return {
        'left' : PrimitiveState(density=1.0, speed=-2.0, pressure=0.4),
        'right': PrimitiveState(density=1.0, speed=2.0, pressure=0.4),
        'p_star' : 0.00189,
        'u_star' : 0.0,
        'rhoL_star' : 0.02185,
        'rhoR_star' : 0.01285
    }

def test_starstate_RP2(RP2):
    """Test computation of Star state for Riemann Problem 2"""

    CPG = CaloricallyPerfectGas(gamma=1.4)
    p_star, u_star = StarState(RP2['left'], RP2['right'], CPG).find()

    assert p_star.round(5) == RP2['p_star']
    assert u_star.round(5) == RP2['u_star']

def test_RP2(RP2):
    """Test Riemann Problem 2"""

    Riemann = ExactRiemannSolver(RP2['left'], RP2['right'])

    s = -0.5 / 0.15
    V = Riemann.sample(s)
    assert V.density == RP2['left'].density
    assert V.speed == RP2['left'].speed
    assert V.pressure == RP2['left'].pressure

    s = (0.2 - 0.5) / 0.15
    V = Riemann.sample(s)
    assert V.density.round(5) == 0.40188
    assert V.speed.round(5) == -1.37639
    assert V.pressure.round(5) == 0.11163

    s = 0.0
    V = Riemann.sample(s)
    assert V.density.round(5) == RP2['rhoL_star']
    assert V.speed == RP2['u_star']
    assert V.pressure.round(5) == RP2['p_star']

    s = 0.5 / 0.15
    V = Riemann.sample(s)
    assert V.density == RP2['right'].density
    assert V.speed == RP2['right'].speed
    assert V.pressure == RP2['right'].pressure

@pytest.fixture
def RP3():
    return {
        'left' : PrimitiveState(density=1.0, speed=0.0, pressure=1000.0),
        'right': PrimitiveState(density=1.0, speed=0.0, pressure=0.01),
        'p_star' : 460.89379,
        'u_star' : 19.59745,
        'rhoL_star' : 0.57506,
        'rhoR_star' : 5.99924
    }

def test_starstate_RP3(RP3):
    """Test computation of Star state for Riemann Problem 3"""
    CPG = CaloricallyPerfectGas(gamma=1.4)
    p_star, u_star = StarState(RP3['left'], RP3['right'], CPG).find()

    assert p_star.round(5) == RP3['p_star']
    assert u_star.round(5) == RP3['u_star']

def test_RP3(RP3):
    """Test Riemann Problem 3"""
    Riemann = ExactRiemannSolver(RP3['left'], RP3['right'])

    s = -0.5 / 0.012
    V = Riemann.sample(s)
    assert V.density == RP3['left'].density
    assert V.speed == RP3['left'].speed
    assert V.pressure == RP3['left'].pressure

    s = -0.1 / 0.012
    V = Riemann.sample(s)
    assert V.density.round(5) == RP3['rhoL_star']
    assert V.speed.round(5) == RP3['u_star']
    assert V.pressure.round(5) == RP3['p_star']

    s = (0.75 - 0.5) / 0.012
    V = Riemann.sample(s)
    assert V.density.round(5) == RP3['rhoR_star']
    assert V.speed.round(5) == RP3['u_star']
    assert V.pressure.round(5) == RP3['p_star']

    s = 0.0
    V = Riemann.sample(s)
    assert V.density.round(5) == RP3['rhoL_star']
    assert V.speed.round(5) == RP3['u_star']
    assert V.pressure.round(5) == RP3['p_star']

    s = 0.5 / 0.012
    V = Riemann.sample(s)
    assert V.density == RP3['right'].density
    assert V.speed == RP3['right'].speed
    assert V.pressure == RP3['right'].pressure

@pytest.fixture
def RP4():
    return {
        'left' : PrimitiveState(density=1.0, speed=0.0, pressure=0.01),
        'right': PrimitiveState(density=1.0, speed=0.0, pressure=100.0),
        'p_star' : 46.09504,
        'u_star' : -6.19633,
        'rhoL_star' : 5.99242,
        'rhoR_star' : 0.57511
    }

def test_starstate_RP4(RP4):
    """Test computation of Star state for Riemann Problem 4"""
    CPG = CaloricallyPerfectGas(gamma=1.4)
    p_star, u_star = StarState(RP4['left'], RP4['right'], CPG).find()

    assert p_star.round(5) == RP4['p_star']
    assert u_star.round(5) == RP4['u_star']

def test_RP4(RP4):
    """Test Riemann Problem 4"""
    Riemann = ExactRiemannSolver(RP4['left'], RP4['right'])

    s = -0.5 / 0.035
    V = Riemann.sample(s)
    assert V.density == RP4['left'].density
    assert V.speed == RP4['left'].speed
    assert V.pressure == RP4['left'].pressure

    s = 0.5 / 0.035
    V = Riemann.sample(s)
    assert V.density == RP4['right'].density
    assert V.speed == RP4['right'].speed
    assert V.pressure == RP4['right'].pressure

@pytest.fixture
def RP5():
    return {
        'left' : PrimitiveState(density=5.99924, speed=19.5975, pressure=460.894),
        'right': PrimitiveState(density=5.99242, speed=-6.19633, pressure=46.095),
        'p_star' : 1691.64696,
        'u_star' : 8.68977,
        'rhoL_star' : 14.2823,
        'rhoR_star' : 31.0426
    }

def test_starstate_RP5(RP5):
    """Test computation of Star state for Riemann Problem 2"""
    CPG = CaloricallyPerfectGas(gamma=1.4)
    p_star, u_star = StarState(RP5['left'], RP5['right'], CPG).find()

    assert p_star.round(5) == RP5['p_star']
    assert u_star.round(5) == RP5['u_star']

def test_RP5(RP5):
    """Test Riemann Problem 5"""
    Riemann = ExactRiemannSolver(RP5['left'], RP5['right'])

    s = -0.5 / 0.035
    V = Riemann.sample(s)
    assert V.density == RP5['left'].density
    assert V.speed == RP5['left'].speed
    assert V.pressure == RP5['left'].pressure

    s = 0.5 / 0.035
    V = Riemann.sample(s)
    assert V.density == RP5['right'].density
    assert V.speed == RP5['right'].speed
    assert V.pressure == RP5['right'].pressure