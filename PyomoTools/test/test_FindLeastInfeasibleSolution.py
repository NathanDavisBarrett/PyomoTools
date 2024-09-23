import pyomo.environ as pyo
import numpy as np

from ..FindLeastInfeasibleSolution import FindLeastInfeasibleSolution
from ..Solvers import DefaultSolver

def test_SimpleProblem():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0,None))

    model.c1 = pyo.Constraint(expr=model.x <= -1)

    FindLeastInfeasibleSolution(model,DefaultSolver("LP"),tee=True)

    xVal = pyo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001

def test_FeasibleProblem():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-1,0))

    FindLeastInfeasibleSolution(model,DefaultSolver("LP"),tee=True)

    xVal = pyo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001

def test_Indexed():
    model = pyo.ConcreteModel()
    model.Set1 = pyo.Set(initialize=["A","B","C"])
    model.x = pyo.Var(model.Set1,bounds=(-1,1))
    model.y = pyo.Var(bounds=(1,2))

    model.c1 = pyo.Constraint(model.Set1,rule=lambda _,i: model.x[i] == model.y)
    model.c2 = pyo.Constraint(expr=model.y == sum(model.x[i] for i in model.Set1))

    FindLeastInfeasibleSolution(model,DefaultSolver("LP"),tee=True)

    results = [pyo.value(model.y),*[pyo.value(model.x[i])for i in model.Set1]]
    assert np.allclose(results,np.zeros(len(results)))
