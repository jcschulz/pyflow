"""
Shared test fixtures.
"""
import pytest
from pyflow.euler.state import PrimitiveState

@pytest.fixture
def riemann_problem_test1():
    return {
        'left' : PrimitiveState(rho=1.0, u=0.0, p=1.0),
        'right': PrimitiveState(rho=0.125, u=0.0, p=0.1),
        'p_star' : 0.30313,
        'u_star' : 0.92745,
        'rhoL_star' : 0.42632,
        'rhoR_star' : 0.26557
    }

@pytest.fixture
def riemann_problem_test2():
    return {
        'left' : PrimitiveState(rho=1.0, u=-2.0, p=0.4),
        'right': PrimitiveState(rho=1.0, u=2.0, p=0.4),
        'p_star' : 0.00189,
        'u_star' : 0.0,
        'rhoL_star' : 0.02185,
        'rhoR_star' : 0.01285
    }

@pytest.fixture
def riemann_problem_test3():
    return {
        'left' : PrimitiveState(rho=1.0, u=0.0, p=1000.0),
        'right': PrimitiveState(rho=1.0, u=0.0, p=0.01),
        'p_star' : 460.89379,
        'u_star' : 19.59745,
        'rhoL_star' : 0.57506,
        'rhoR_star' : 5.99924
    }

@pytest.fixture
def riemann_problem_test4():
    return {
        'left' : PrimitiveState(rho=1.0, u=0.0, p=0.01),
        'right': PrimitiveState(rho=1.0, u=0.0, p=100.0),
        'p_star' : 46.09504,
        'u_star' : -6.19633,
        'rhoL_star' : 5.99242,
        'rhoR_star' : 0.57511
    }

@pytest.fixture
def riemann_problem_test5():
    return {
        'left' : PrimitiveState(rho=5.99924, u=19.5975, p=460.894),
        'right': PrimitiveState(rho=5.99242, u=-6.19633, p=46.095),
        'p_star' : 1691.64696,
        'u_star' : 8.68977,
        'rhoL_star' : 14.2823,
        'rhoR_star' : 31.0426
    }
