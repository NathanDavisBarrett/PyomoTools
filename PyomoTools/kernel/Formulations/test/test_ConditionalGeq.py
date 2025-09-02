import pyomo.kernel as pmo
import numpy as np

from ..ConditionalGeq import ConditionalGeq
from ....base.Solvers import DefaultSolver


def test_Construction():
    Amin = -2
    Amax = 10

    m = pmo.block()
    m.A = pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax)
    alpha = 5
    m.C = ConditionalGeq(m.A, alpha)
    # m.C.Plot()


def test_leq():
    Amin = -2
    Amax = 10

    m = pmo.block()
    m.A = pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax)
    alpha = 5
    m.C = ConditionalGeq(m.A, alpha)

    m.A.fix(3)

    m.obj = pmo.objective(m.C.X, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert np.allclose(pmo.value(m.C.X), 0)


def test_geq():
    Amin = -2
    Amax = 10

    m = pmo.block()
    m.A = pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax)
    alpha = 5
    m.C = ConditionalGeq(m.A, alpha)

    m.A.fix(7)

    m.obj = pmo.objective(m.C.X, sense=pmo.minimize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert np.allclose(pmo.value(m.C.X), 1)


def test_middle():
    Amin = -2
    Amax = 10

    m = pmo.block()
    m.A = pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax)
    alpha = 5
    m.C = ConditionalGeq(m.A, alpha, epsilon=2)

    m.A.fix(4.5)

    m.obj = pmo.objective(m.C.X, sense=pmo.minimize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.infeasible


def test_minimum():
    Amin = -2
    Amax = 10

    m = pmo.block()
    m.A = pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax)
    alpha = -2
    m.C = ConditionalGeq(m.A, alpha)

    m.A.fix(4.5)

    m.obj = pmo.objective(m.C.X, sense=pmo.minimize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert np.allclose(pmo.value(m.C.X), 1)


def test_maximum():
    Amin = -2
    Amax = 10

    m = pmo.block()
    m.A = pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax)
    alpha = 10
    m.C = ConditionalGeq(m.A, alpha)

    m.A.fix(4.5)

    m.obj = pmo.objective(m.C.X, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert np.allclose(pmo.value(m.C.X), 0)
