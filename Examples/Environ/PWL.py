"""
A example workflow how to include a "PWL" constraint in a model.

This model can be represented as follows:

max Y
s.t. Y = 5 sin(X - pi/2)
     0 <= X <= 2 pi

Where X and Y are continuous variables and Z is a binary variable.
"""

import pyomo.environ as pyo
import numpy as np
from PyomoTools.environ.Formulations import PWL
from PyomoTools.base.Solvers import DefaultSolver
from PyomoTools.environ.IO import ModelToExcel


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
    args=(
        myMag,
    ),  # Note that I could have passed these both as args or both as kwargs.
    kwargs={
        "shift": myShift
    },  # ...................................................................
    verify=True,
)

model.obj = pyo.Objective(expr=model.Y, sense=pyo.maximize)
solver = DefaultSolver("MILP")
solver.solve(model)

ModelToExcel(model, "PWLSolution.xlsx")
