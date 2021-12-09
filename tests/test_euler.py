import numpy as np


from pyflow.euler.model import Euler, CaloricallyPerfectGas

def test_cpg_speed_of_sound():

    CPG = CaloricallyPerfectGas()

    assert CPG.speed_of_sound(1.0, 1.0).round(4) == np.sqrt(1.4).round(4)
    assert (CPG.speed_of_sound(np.ones((10,)), np.ones((10,))).round(4) == np.sqrt(1.4 * np.ones((10,))).round(4)).all


def test_cpg_cons2prim():

    model = Euler(CaloricallyPerfectGas())
    U = np.zeros((3,1))
    U[0] = 1.0
    U[1] = 0.0
    U[2] = 1.0
    V = model.conservatives_to_primitives(U)
    assert V[0,0] == 1.0
    assert V[1,0] == 0.0
    assert V[2,0].round(2) == round(0.4,2)


def test_cpg_prim2cons():

    model = Euler(CaloricallyPerfectGas())
    V = np.zeros((3,1))
    V[0] = 1.0
    V[1] = 2.0
    V[2] = 3.0
    U = model.primitives_to_conservatives(V)
    assert U[0,0] == 1.0
    assert U[1,0] == 2.0
    assert U[2,0].round(3) == round(3.0 / 0.4 + 2.0, 3)
