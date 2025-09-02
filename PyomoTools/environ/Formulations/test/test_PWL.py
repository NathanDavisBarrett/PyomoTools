import pyomo.environ as pyo
from ..PWL import PWL
from ....base.Solvers import DefaultSolver

import numpy as np


def test_functionGenerator():
    def myFunc(x, magnitude, shift):
        return magnitude * np.sin(x - shift)

    myMag = 5
    myShift = np.pi / 2
    xBounds = (0, 2 * np.pi)

    model = pyo.ConcreteModel()
    model.X = pyo.Var(bounds=xBounds)
    model.Y = pyo.Var()

    PWL(
        model=model,
        func=myFunc,
        xVar=model.X,
        yVar=model.Y,
        xBounds=xBounds,
        numSegments=6,
        args=(myMag,),
        kwargs={"shift": myShift},
        verify=False,
    )

    model.obj = pyo.Objective(expr=model.Y, sense=pyo.maximize)
    solver = DefaultSolver("MILP")

    textX = np.linspace(*xBounds, 10)

    for x in textX:
        model.X.fix(x)
        solver.solve(model)
        yPred = myFunc(x, myMag, myShift)

        assert np.allclose(
            [pyo.value(model.Y)],
            [
                yPred,
            ],
            atol=0.1 * myMag,
        )


def test_arrayGenerator():
    def myFunc(x, magnitude, shift):
        return magnitude * np.sin(x - shift)

    myMag = 5
    myShift = np.pi / 2
    xBounds = (0, 2 * np.pi)

    xs = np.linspace(*xBounds, 20)
    ys = myFunc(xs, myMag, myShift)
    points = np.vstack([xs, ys]).T

    model = pyo.ConcreteModel()
    model.X = pyo.Var(bounds=xBounds)
    model.Y = pyo.Var()

    PWL(
        model=model,
        func=points,
        xVar=model.X,
        yVar=model.Y,
        xBounds=xBounds,
        numSegments=6,
        args=(myMag,),
        kwargs={"shift": myShift},
        verify=False,
    )

    model.obj = pyo.Objective(expr=model.Y, sense=pyo.maximize)
    solver = DefaultSolver("MILP")

    textX = np.linspace(*xBounds, 10)

    for x in textX:
        model.X.fix(x)
        solver.solve(model)
        yPred = myFunc(x, myMag, myShift)

        assert np.allclose(
            [pyo.value(model.Y)],
            [
                yPred,
            ],
            atol=0.1 * myMag,
        )
