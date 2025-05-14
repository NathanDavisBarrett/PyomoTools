"""
A example workflow how/when to use the LoadIndexedSet function

In particular, this example demonstrates how to solve basic route planning problem
"""

import pyomo.environ as pyo
from PyomoTools import LoadIndexedSet
from PyomoTools.IO import ModelToExcel
from PyomoTools.base.Solvers import DefaultSolver

model = pyo.ConcreteModel()

######## PARAMETERS ########
Points = ["A","B","C","D"]
Connections = ["AB","AC","BC","BD","CD"]
TerminalPoints = ["A","D"]

delta = {
    "AB": 15,
    "AC": 5,
    "BC": 4,
    "BD": 2,
    "CD": 10
}

######## PYOMO SETS ########
model.setP = pyo.Set(initialize=Points)
model.setP_Terminal = pyo.Set(initialize=TerminalPoints)
model.setP_NONTerminal = model.setP - model.setP_Terminal

model.setC = pyo.Set(initialize=Connections)
LoadIndexedSet(
    model,
    "setC_p",
    {p: [c for c in model.setC if p in c] for p in model.setP}
)


######## VARIABLES ########
model.ExecuteTransition = pyo.Var(model.setC,domain=pyo.Binary)

model.VisitPoint = pyo.Var(model.setP,domain=pyo.Binary)

######## CONSTRAINTS ########
def NonTerminalTravelConstraint(model,p):
    return sum(model.ExecuteTransition[c] for c in model.setC_p[p]) == 2 * model.VisitPoint[p]
model.NonTerminalTravelConstraint = pyo.Constraint(model.setP_NONTerminal,rule=NonTerminalTravelConstraint)

def TerminalTravelConstraint(model,p):
    return sum(model.ExecuteTransition[c] for c in model.setC_p[p]) == model.VisitPoint[p]
model.TerminalTravelConstraint = pyo.Constraint(model.setP_Terminal,rule=TerminalTravelConstraint)

def TerminalMandate(model,p):
    return model.VisitPoint[p] == 1
model.TerminalMandate = pyo.Constraint(model.setP_Terminal,rule=TerminalMandate)

######## OBJECTIVE ########
model.Obj = pyo.Objective(expr=sum(delta[c] * model.ExecuteTransition[c] for c in model.setC))

######## SOLVE ########
solver = DefaultSolver("MILP")
solver.solve(model)

######## SAVE RESULTS ########
ModelToExcel(model,"RoutePlanning.xlsx")