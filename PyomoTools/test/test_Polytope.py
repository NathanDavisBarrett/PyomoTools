import pyomo.environ as pyo
import numpy as np

from ..Polytope import Polytope
from ..Solvers import DefaultSolver

def test_2D():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-5,5))
    model.y = pyo.Var(bounds=(-5,5))
    model.z = pyo.Var(bounds=(-5,5))
    model.a = pyo.Var(bounds=(-100,100))

    model.c = pyo.Constraint(expr=3*(5*model.x + model.y - 13) <= model.z/2 + 10 + model.a)

    polytope = Polytope(model,[model.x,model.y])
    polytope.Plot()

def test_3D():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-5,5))
    model.y = pyo.Var(bounds=(-5,5))
    model.z = pyo.Var(bounds=(-5,5))
    model.a = pyo.Var(bounds=(-100,100))

    model.c = pyo.Constraint(expr=3*(5*model.x + model.y - 13) <= model.z/2 + 10 + model.a)

    polytope = Polytope(model,[model.x,model.y,model.z])
    polytope.Plot()

def test_2D_DropConstr():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-5,5))
    model.y = pyo.Var(bounds=(-5,5))
    model.z = pyo.Var(bounds=(-5,5))
    model.a = pyo.Var(bounds=(-100,100))

    model.c = pyo.Constraint(expr=3*(5*model.x + model.y - 13) <= model.z/2 + 10 + model.a)

    model.c2 = pyo.Constraint(expr=model.z + model.a == 1)

    polytope = Polytope(model,[model.x,model.y])
    polytope.Plot()

