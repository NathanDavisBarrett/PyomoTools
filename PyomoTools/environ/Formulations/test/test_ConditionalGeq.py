import pyomo.environ as pyo
import numpy as np

from ..ConditionalGeq import ConditionalGeq
from ....base.Solvers import DefaultSolver


def test_Construction():
    Amin = -2
    Amax = 10

    m = pyo.ConcreteModel()
    m.A = pyo.Var(domain=pyo.Reals, bounds=(Amin, Amax))
    alpha = 5
    upperBound, lowerBound, X = ConditionalGeq(m, m.A, alpha, A_bounds=(Amin, Amax))

    # Check that constraints were created
    assert hasattr(m, "A_ConditionalGeq_upperBound")
    assert hasattr(m, "A_ConditionalGeq_lowerBound")
    assert hasattr(m, "A_ConditionalGeq_X")


def test_leq():
    Amin = -2
    Amax = 10

    m = pyo.ConcreteModel()
    m.A = pyo.Var(domain=pyo.Reals, bounds=(Amin, Amax))
    alpha = 5
    upperBound, lowerBound, X = ConditionalGeq(m, m.A, alpha, A_bounds=(Amin, Amax))

    m.A.fix(3)

    m.obj = pyo.Objective(expr=X, sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert np.allclose(pyo.value(X), 0)


def test_geq():
    Amin = -2
    Amax = 10

    m = pyo.ConcreteModel()
    m.A = pyo.Var(domain=pyo.Reals, bounds=(Amin, Amax))
    alpha = 5
    upperBound, lowerBound, X = ConditionalGeq(m, m.A, alpha, A_bounds=(Amin, Amax))

    m.A.fix(7)

    m.obj = pyo.Objective(expr=X, sense=pyo.minimize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert np.allclose(pyo.value(X), 1)


def test_middle():
    Amin = -2
    Amax = 10

    m = pyo.ConcreteModel()
    m.A = pyo.Var(domain=pyo.Reals, bounds=(Amin, Amax))
    alpha = 5
    upperBound, lowerBound, X = ConditionalGeq(
        m, m.A, alpha, epsilon=2, A_bounds=(Amin, Amax)
    )

    m.A.fix(4.5)

    m.obj = pyo.Objective(expr=X, sense=pyo.minimize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_minimum():
    Amin = -2
    Amax = 10

    m = pyo.ConcreteModel()
    m.A = pyo.Var(domain=pyo.Reals, bounds=(Amin, Amax))
    alpha = -2
    upperBound, lowerBound, X = ConditionalGeq(m, m.A, alpha, A_bounds=(Amin, Amax))

    m.A.fix(4.5)

    m.obj = pyo.Objective(expr=X, sense=pyo.minimize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert np.allclose(pyo.value(X), 1)


def test_maximum():
    Amin = -2
    Amax = 10

    m = pyo.ConcreteModel()
    m.A = pyo.Var(domain=pyo.Reals, bounds=(Amin, Amax))
    alpha = 10
    upperBound, lowerBound, X = ConditionalGeq(m, m.A, alpha, A_bounds=(Amin, Amax))

    m.A.fix(4.5)

    m.obj = pyo.Objective(expr=X, sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert np.allclose(pyo.value(X), 0)
