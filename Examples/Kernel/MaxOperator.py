"""
A example workflow how to include a "DoubleSidedBigM" constraint in a model.

This model can be represented as follows:

max A
s.t. A = max(B,C)
     2 <= B <= 10
     -50 <= C <= 100
"""

import pyomo.kernel as pmo
from PyomoTools.kernel.Formulations import MaxOperator
from PyomoTools.base.Solvers import DefaultSolver
from PyomoTools.kernel.IO import ModelToYaml

solver = DefaultSolver("MILP")
model = pmo.block()

bBounds = (2, 10)
cBounds = (-50, 100)

model.A = pmo.variable()
model.B = pmo.variable(lb=bBounds[0], ub=bBounds[1])
model.C = pmo.variable(lb=cBounds[0], ub=cBounds[1])

model.MaxOper = MaxOperator(
    A=model.A,
    B=model.B,
    C=model.C,
    bBounds=bBounds,
    cBounds=cBounds,
)

model.obj = pmo.objective(expr=model.A, sense=pmo.maximize)

solver.solve(model)

ModelToYaml(model, "MaxOperatorSolution.yaml")
