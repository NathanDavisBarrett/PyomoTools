"""
A example workflow how to include a "DoubleSidedBigM" constraint in a model.

This model can be represented as follows:

max A
s.t. A = max(B,C)
     2 <= B <= 10
     -50 <= C <= 100
"""

import pyomo.environ as pyo
from PyomoTools.environ.Formulations import MaxOperator
from PyomoTools.base.Solvers import DefaultSolver
from PyomoTools.environ.IO import ModelToExcel

solver = DefaultSolver("MILP")
model = pyo.ConcreteModel()

bBounds = (2, 10)
cBounds = (-50, 100)

model.A = pyo.Var()
model.B = pyo.Var(bounds=bBounds)
model.C = pyo.Var(bounds=cBounds)

MaxOperator(
    model=model,
    A=model.A,
    B=model.B,
    C=model.C,
    bBounds=bBounds,
    cBounds=cBounds,
)

model.obj = pyo.Objective(expr=model.A, sense=pyo.maximize)

solver.solve(model)

ModelToExcel(model, "MaxOperatorSolution.xlsx")
