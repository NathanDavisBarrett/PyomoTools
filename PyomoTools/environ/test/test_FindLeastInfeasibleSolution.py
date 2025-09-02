import pyomo.environ as pyo
import numpy as np

from ..FindLeastInfeasibleSolution import (
    FindLeastInfeasibleSolution,
    LeastInfeasibleDefinition,
)
from ...base.Solvers import DefaultSolver


def test_SimpleProblem1():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, None))

    model.c1 = pyo.Constraint(expr=model.x <= -1)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pyo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_SimpleProblem_KnownSolution():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    model.c1 = pyo.Constraint(expr=model.y >= 2)
    model.c2 = pyo.Constraint(expr=model.y >= -model.x + 4)
    model.c3 = pyo.Constraint(expr=model.y <= -model.x + 2)
    model.c4 = pyo.Constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(model, DefaultSolver("QP"), tee=True)

    # Any point on the line y = x in 1 <= x <= 2 is a valid solution.

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    assert np.allclose(
        [
            xVal,
        ],
        [
            yVal,
        ],
    )
    assert xVal >= -0.9999999
    assert xVal <= 2.0000001


def test_L2():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    model.c1 = pyo.Constraint(expr=model.y >= 2)
    model.c2 = pyo.Constraint(expr=model.y >= -model.x + 4)
    model.c3 = pyo.Constraint(expr=model.y <= -model.x + 2)
    model.c4 = pyo.Constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("QP"),
        leastInfeasibleDefinition=LeastInfeasibleDefinition.L2_Norm,
    )

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    assert np.allclose([xVal, yVal], [1.5, 1.5])


def test_FeasibleProblem():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-1, 0))

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pyo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_Indexed():
    model = pyo.ConcreteModel()
    model.Set1 = pyo.Set(initialize=["A", "B", "C"])
    model.x = pyo.Var(model.Set1, bounds=(-1, 1))
    model.y = pyo.Var(bounds=(1, 2))

    model.c1 = pyo.Constraint(model.Set1, rule=lambda _, i: model.x[i] == model.y)
    model.c2 = pyo.Constraint(expr=model.y == sum(model.x[i] for i in model.Set1))

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    results = [pyo.value(model.y), *[pyo.value(model.x[i]) for i in model.Set1]]
    assert np.allclose(results, np.zeros(len(results)))
