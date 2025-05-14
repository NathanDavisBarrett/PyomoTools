"""
A example workflow how to use the InfeasibilityReport class
"""

import pyomo.environ as pyo
from PyomoTools import InfeasibilityReport

model = pyo.ConcreteModel()

model.X = pyo.Var()
model.Y = pyo.Var()

model.Constraint1 = pyo.Constraint(expr=model.X <= model.Y)
model.Constraint2 = pyo.Constraint(expr=model.X >= model.Y + 1)
model.Constraint3 = pyo.Constraint(expr=model.X <= 11)
model.Constraint4 = pyo.Constraint(expr=model.X >= 10)

#Alternatively, you could load values from a potential solution excel file using IO.LoadModelSolutionFromExcel
model.X.value = 1
model.Y.value = 2

report = InfeasibilityReport(model)
print("The following constraints are violated:")
print(report)