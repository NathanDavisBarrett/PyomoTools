"""
A example workflow how to include a "DoubleSidedBigM" constraint in a model.

This model can be represented as follows:

max Y
s.t. Y = X * Z
     -2 <= x <= 10

Where X and Y are continuous variables and Z is a binary variable.
"""

import pyomo.kernel as pmo
from PyomoTools.kernel.Formulations import DoubleSidedBigM
from PyomoTools.base.Solvers import DefaultSolver
from PyomoTools.kernel.IO import ModelToYaml

xBounds = [-2,10]

model = pmo.block()
model.Y = pmo.variable(domain=pmo.Reals)
model.X = pmo.variable(domain=pmo.Reals)
model.Z = pmo.variable(domain=pmo.Binary)

model.DSBM = DoubleSidedBigM(
    A=model.Y,
    B=model.X,
    X=model.Z,
    Bmin=xBounds[0],
    Bmax=xBounds[1]
)

model.obj = pmo.objective(expr=model.Y,sense=pmo.maximize)

solver = DefaultSolver("MILP")
solver.solve(model)

ModelToYaml(model,"DoubleSidedBigMSolution.yaml")
