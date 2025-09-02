import pyomo.kernel as pmo
import numpy as np

from ..FindLeastInfeasibleSolution import (
    FindLeastInfeasibleSolution,
    LeastInfeasibleDefinition,
)
from ...base.Solvers import DefaultSolver


def test_SimpleProblem1():
    model = pmo.block()
    model.x = pmo.variable(lb=0)

    model.c1 = pmo.constraint(model.x <= -1)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pmo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_SimpleProblem_KnownSolution():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    model.c1 = pmo.constraint(expr=model.y >= 2)
    model.c2 = pmo.constraint(expr=model.y >= -model.x + 4)
    model.c3 = pmo.constraint(expr=model.y <= -model.x + 2)
    model.c4 = pmo.constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("LP"),
        tee=True,
        leastInfeasibleDefinition=LeastInfeasibleDefinition.L2_Norm,
    )

    # Any point on the line y = x in 1 <= x <= 2 is a valid solution.

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.y)

    assert np.allclose([xVal, yVal], [1.5, 1.5])


def test_L2():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    model.c1 = pmo.constraint(expr=model.y >= 2)
    model.c2 = pmo.constraint(expr=model.y >= -model.x + 4)
    model.c3 = pmo.constraint(expr=model.y <= -model.x + 2)
    model.c4 = pmo.constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("QP"),
        leastInfeasibleDefinition=LeastInfeasibleDefinition.L2_Norm,
    )

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.y)

    assert np.allclose([xVal, yVal], [1.5, 1.5])


def test_FeasibleProblem():
    model = pmo.block()
    model.x = pmo.variable(lb=-1, ub=0)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pmo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_Indexed():
    model = pmo.block()
    model.x = pmo.variable_list([pmo.variable(lb=0) for i in range(3)])

    model.c1 = pmo.constraint_list([pmo.constraint(model.x[i] <= -1) for i in range(3)])

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    for i in range(3):
        xVal = pmo.value(model.x[i])
        assert xVal >= -1.000001
        assert xVal <= 0.000001


def test_Multilevel():
    model = pmo.block()
    model.x = pmo.variable(lb=0)
    model.sub = pmo.block()
    model.sub.x = pmo.variable(lb=0)

    model.c1 = pmo.constraint(model.x <= -1)
    model.sub.c1 = pmo.constraint(model.sub.x <= -1)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)
    xVal = pmo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001

    xVal = pmo.value(model.sub.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001
