
from pyflow.euler.exact_riemann import ExactRiemannSolver

def test_starstate_RP1(riemann_problem_test1):
    """Test computation of Star state for Riemann Problem 1"""
    RP = ExactRiemannSolver()
    RP.initialize(
        riemann_problem_test1['left'],
        riemann_problem_test1['right'],
    )
    p_star, u_star = RP.star_state()

    assert p_star.round(5) == riemann_problem_test1['p_star']
    assert u_star.round(5) == riemann_problem_test1['u_star']

def test_starstate_RP2(riemann_problem_test2):
    """Test Riemann Problem 2"""
    RP = ExactRiemannSolver()
    RP.initialize(
        riemann_problem_test2['left'],
        riemann_problem_test2['right'],
    )
    p_star, u_star = RP.star_state()

    assert p_star.round(5) == riemann_problem_test2['p_star']
    assert u_star.round(5) == riemann_problem_test2['u_star']

def test_starstate_RP3(riemann_problem_test3):
    """Test Riemann Problem 3"""
    RP = ExactRiemannSolver()
    RP.initialize(
        riemann_problem_test3['left'],
        riemann_problem_test3['right'],
    )
    p_star, u_star = RP.star_state()

    assert p_star.round(5) == riemann_problem_test3['p_star']
    assert u_star.round(5) == riemann_problem_test3['u_star']

def test_starstate_RP4(riemann_problem_test4):
    """Test Riemann Problem 4"""
    RP = ExactRiemannSolver()
    RP.initialize(
        riemann_problem_test4['left'],
        riemann_problem_test4['right'],
    )
    p_star, u_star = RP.star_state()

    assert p_star.round(5) == riemann_problem_test4['p_star']
    assert u_star.round(5) == riemann_problem_test4['u_star']

def test_starstate_RP5(riemann_problem_test5):
    """Test Riemann Problem 5"""
    RP = ExactRiemannSolver()
    RP.initialize(
        riemann_problem_test5['left'],
        riemann_problem_test5['right'],
    )
    p_star, u_star = RP.star_state()

    assert p_star.round(5) == riemann_problem_test5['p_star']
    assert u_star.round(5) == riemann_problem_test5['u_star']
