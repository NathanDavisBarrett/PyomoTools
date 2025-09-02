"""
A example workflow how to include a "DoubleSidedBigM" constraint in a model.

This model can be represented as follows:

max Y
s.t. Y = X * Z
     -2 <= x <= 10

Where X and Y are continuous variables and Z is a binary variable.
"""

import pyomo.environ as pyo
from PyomoTools.environ.Formulations import DoubleSidedBigM
from PyomoTools.base.Solvers import DefaultSolver
from PyomoTools.environ.IO import ModelToExcel

xBounds = [-2, 10]

model = pyo.ConcreteModel()
model.Y = pyo.Var(domain=pyo.Reals)
model.X = pyo.Var(domain=pyo.Reals)
model.Z = pyo.Var(domain=pyo.Binary)

DoubleSidedBigM(
    model=model, A=model.Y, B=model.X, X=model.Z, Bmin=xBounds[0], Bmax=xBounds[1]
)

model.obj = pyo.Objective(expr=model.Y, sense=pyo.maximize)

solver = DefaultSolver("MILP")
solver.solve(model)

ModelToExcel(model, "DoubleSidedBigMSolution.xlsx")
