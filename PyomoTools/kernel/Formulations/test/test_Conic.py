import pyomo.kernel as pmo
import numpy as np

from ..Conic import Conic
from ....base.Solvers import DefaultSolver


def test_even_order():
    Amin = -5
    Amax = 5
    order = 2

    m = pmo.block()
    m.junk = pmo.variable(domain=pmo.NonNegativeReals)
    m.x = [pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax) for _ in range(3)]
    m.r = pmo.variable(domain=pmo.NonNegativeReals)
    m.C = Conic(m.x, m.r, order)

    # Feasible: sum(x_i^2) <= r^2
    for i, v in enumerate([1.0, 2.0, 1.0]):
        m.x[i].fix(v)
    m.r.fix(3.0)
    m.obj = pmo.objective(m.junk, sense=pmo.minimize)
    solver = DefaultSolver("NLP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert np.allclose(pmo.value(m.r), 3.0)

    # Infeasible: sum(x_i^2) > r^2
    for i, v in enumerate([2.0, 2.0, 2.0]):
        m.x[i].fix(v)
    m.r.fix(2.0)
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.infeasible


def test_odd_order():
    Amin = -5
    Amax = 5
    order = 3

    m = pmo.block()
    m.junk = pmo.variable(domain=pmo.NonNegativeReals)
    m.x = [pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax) for _ in range(2)]
    m.r = pmo.variable(domain=pmo.NonNegativeReals)
    m.C = Conic(m.x, m.r, order)

    # Feasible: sum(x_i^3) <= r^3 and sum(-x_i^3) <= r^3
    m.x[0].fix(1.0)
    m.x[1].fix(-1.0)
    m.r.fix(2.0)
    m.obj = pmo.objective(m.junk, sense=pmo.minimize)
    solver = DefaultSolver("NLP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert np.allclose(pmo.value(m.r), 2.0)

    # Infeasible: sum(x_i^3) > r^3 or sum(-x_i^3) > r^3
    m.x[0].fix(3.0)
    m.x[1].fix(3.0)
    m.r.fix(1.0)
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.infeasible


def test_explicit_r_nonnegativity():
    Amin = -5
    Amax = 5
    order = 3

    m = pmo.block()
    m.junk = pmo.variable(domain=pmo.NonNegativeReals)
    m.x = [pmo.variable(domain=pmo.Reals, lb=Amin, ub=Amax) for _ in range(2)]
    m.r = pmo.variable(domain=pmo.Reals, lb=-10, ub=10)
    m.C = Conic(m.x, m.r, order, explicit_r_nonnegativity=True)

    # r < 0 should be infeasible
    m.x[0].fix(0.0)
    m.x[1].fix(0.0)
    m.r.fix(-1.0)
    m.obj = pmo.objective(m.junk, sense=pmo.minimize)
    solver = DefaultSolver("NLP")
    result = solver.solve(m)
    assert result.solver.termination_condition == pmo.TerminationCondition.infeasible
