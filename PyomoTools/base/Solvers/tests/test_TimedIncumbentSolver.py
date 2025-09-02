from ..TimedIncumbentSolver import TimedIncumbentSolver

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np


def test_Simple():

    # Pyomo model
    model = pyo.ConcreteModel()
    model.x = pyo.Var(within=pyo.Binary)
    model.y = pyo.Var(within=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=2 * model.x + model.y, sense=pyo.maximize)
    model.c1 = pyo.Constraint(expr=model.x + model.y <= 1.5)

    solver = TimedIncumbentSolver(5)
    results = solver.solve(model)

    assert results.solver.termination_condition == TerminationCondition.optimal

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    assert np.allclose(np.array([xVal, yVal]), np.array([1.0, 0.5]))


def test_FindFirstIncumbent():

    # Pyomo model
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(1, 1000)
    model.x = pyo.Var(model.I, within=pyo.Binary)
    model.y = pyo.Var(model.I, within=pyo.NonNegativeReals)
    model.obj = pyo.Objective(
        expr=sum(2 * model.x[i] + model.y[i] for i in model.I), sense=pyo.maximize
    )
    model.c1 = pyo.ConstraintList()
    for i in model.I:
        model.c1.add(expr=model.x[i] + model.y[i] <= 1.5)
    model.c2 = pyo.Constraint(expr=sum(model.x[i] for i in model.I) <= 500)

    solver = TimedIncumbentSolver(0)
    solver.solve(model, tee=True)

    # assert that a solution was found:
    #   This will only happen if a value is loaded into model.x[2]
    val = pyo.value(model.x[2])
    assert val is not None
