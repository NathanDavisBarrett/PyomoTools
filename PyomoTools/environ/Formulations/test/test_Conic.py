import pyomo.environ as pyo
import numpy as np

from ..Conic import Conic
from ....base.Solvers import DefaultSolver


def test_even_order():
    Amin = -5
    Amax = 5
    order = 2
    m = pyo.ConcreteModel()
    m.junk = pyo.Var(domain=pyo.NonNegativeReals)
    m.x = pyo.Var(range(3), domain=pyo.Reals, bounds=(Amin, Amax))
    m.r = pyo.Var(domain=pyo.NonNegativeReals)
    conic_constraint, r_nonnegativity, conic_constraint_neg = Conic(
        m,
        [m.x[i] for i in range(3)],
        m.r,
        order,
        "test_conic",
    )
    # Feasible: sum(x_i^2) <= r^2
    for i, v in enumerate([1.0, 2.0, 1.0]):
        m.x[i].fix(v)
    m.r.fix(3.0)
    m.obj = pyo.Objective(expr=m.junk, sense=pyo.minimize)
    solver = DefaultSolver("NLP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert np.allclose(pyo.value(m.r), 3.0)
    # Infeasible: sum(x_i^2) > r^2
    for i, v in enumerate([2.0, 2.0, 2.0]):
        m.x[i].fix(v)
    m.r.fix(2.0)
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_odd_order():
    Amin = -5
    Amax = 5
    order = 3
    m = pyo.ConcreteModel()
    m.junk = pyo.Var(domain=pyo.NonNegativeReals)
    m.x = pyo.Var(range(2), domain=pyo.Reals, bounds=(Amin, Amax))
    m.r = pyo.Var(domain=pyo.NonNegativeReals)
    conic_constraint, r_nonnegativity, conic_constraint_neg = Conic(
        m, [m.x[i] for i in range(2)], m.r, order, "test_conic"
    )
    # Feasible: sum(x_i^3) <= r^3 and sum(-x_i^3) <= r^3
    m.x[0].fix(1.0)
    m.x[1].fix(-1.0)
    m.r.fix(2.0)
    m.obj = pyo.Objective(expr=m.junk, sense=pyo.minimize)
    solver = DefaultSolver("NLP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert np.allclose(pyo.value(m.r), 2.0)
    # Infeasible: sum(x_i^3) > r^3 or sum(-x_i^3) > r^3
    m.x[0].fix(3.0)
    m.x[1].fix(3.0)
    m.r.fix(1.0)
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_explicit_r_nonnegativity():
    Amin = -5
    Amax = 5
    order = 3
    m = pyo.ConcreteModel()
    m.junk = pyo.Var(domain=pyo.NonNegativeReals)
    m.x = pyo.Var(range(2), domain=pyo.Reals, bounds=(Amin, Amax))
    m.r = pyo.Var(domain=pyo.Reals, bounds=(-10, 10))
    conic_constraint, r_nonnegativity, conic_constraint_neg = Conic(
        m,
        [m.x[i] for i in range(2)],
        m.r,
        order,
        "test_conic",
        explicit_r_nonnegativity=True,
    )
    # r < 0 should be infeasible
    m.x[0].fix(0.0)
    m.x[1].fix(0.0)
    m.r.fix(-1.0)
    m.obj = pyo.Objective(expr=m.junk, sense=pyo.minimize)
    solver = DefaultSolver("NLP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.infeasible
