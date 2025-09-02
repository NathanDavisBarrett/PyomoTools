"""
A example workflow how to use the InfeasibilityReport class
"""

import pyomo.kernel as pmo
from PyomoTools.kernel import InfeasibilityReport

model = pmo.block()

model.X = pmo.variable()
model.Y = pmo.variable()

model.Constraint1 = pmo.constraint(expr=model.X <= model.Y)
model.Constraint2 = pmo.constraint(expr=model.X >= model.Y + 1)
model.Constraint3 = pmo.constraint(expr=model.X <= 11)
model.Constraint4 = pmo.constraint(expr=model.X >= 10)

# Alternatively, you could load values from a potential solution excel file using IO.LoadModelSolutionFromExcel
model.X.value = 1
model.Y.value = 2

report = InfeasibilityReport(model)
print("The following constraints are violated:")
print(report)
report.WriteFile("InfeasibilityReport.txt")
