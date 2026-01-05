import pyomo.kernel as pmo

from ..RelaxIntegerVarsKernel import RelaxIntegerVarsKernel as RelaxIntegrality
from ...base.Solvers import DefaultSolver

import numpy as np


def test_RelaxIntegrality():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Binary)
    model.y = pmo.variable(domain=pmo.NonNegativeReals)

    model.c1 = pmo.constraint(model.x + 2 * model.y <= 2.5)
    model.c2 = pmo.constraint(2 * model.x + model.y <= 1.7)

    model.obj = pmo.objective(expr=model.x + model.y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    results = solver.solve(model, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal
    sol = np.array([model.x.value, model.y.value])
    expected = np.array([0.0, 1.25])
    assert np.allclose(sol, expected)

    transformation = RelaxIntegrality()
    transformation.apply_to(model)

    results = solver.solve(model, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal
    sol = np.array([model.x.value, model.y.value])
    expected = np.array([0.3, 1.1])
    assert np.allclose(sol, expected)


def test_BoundPreservation():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Binary)
    model.y = pmo.variable(domain=pmo.NonNegativeReals)

    model.c1 = pmo.constraint(model.x + 2 * model.y <= 2.5)
    model.c2 = pmo.constraint(2 * model.x + model.y <= 3.0)

    model.obj = pmo.objective(expr=model.x + model.y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    results = solver.solve(model, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal
    sol = np.array([model.x.value, model.y.value])
    expected = np.array([1.0, 0.75])
    assert np.allclose(sol, expected)

    transformation = RelaxIntegrality()
    transformation.apply_to(model)

    results = solver.solve(model, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal
    sol = np.array([model.x.value, model.y.value])
    expected = np.array([1.0, 0.75])
    assert np.allclose(sol, expected)
