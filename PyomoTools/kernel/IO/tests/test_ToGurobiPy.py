import pyomo.kernel as pmo
import gurobipy

import pytest
import numpy as np

from ....base.Solvers import DefaultSolver

from ..ToGurobiPy import ToGurobiPy


@pytest.fixture()
def basic_model():
    m = pmo.block()
    m.x = pmo.variable(lb=0, ub=1)
    m.y = pmo.variable(lb=0, ub=1, domain=pmo.Reals)
    m.z = pmo.variable(lb=0, ub=1)
    m.c1 = pmo.constraint(m.x + m.y <= 1)
    m.c2 = pmo.constraint(m.y + m.z <= 1)
    m.c3 = pmo.constraint(m.x + m.z <= 1)
    m.o = pmo.objective(m.x + m.y + m.z)
    return m


def test_Construction(basic_model):
    gurobi_model = ToGurobiPy(basic_model)

    gurobi_model.update()

    assert isinstance(gurobi_model, gurobipy.Model)
    assert gurobi_model.NumVars == 3
    assert gurobi_model.NumConstrs == 3
    assert gurobi_model.NumSOS == 0
    assert gurobi_model.NumQConstrs == 0
    assert gurobi_model.NumGenConstrs == 0


def test_CorrectAnswer(basic_model):
    gurobi_model = ToGurobiPy(basic_model)

    solver = DefaultSolver("LP")
    solver.solve(basic_model)

    pyomoAnswers = np.array(
        [basic_model.x.value, basic_model.y.value, basic_model.z.value]
    )

    gurobi_model.optimize()
    gurobiAnswers = np.array(
        [
            gurobi_model.getVarByName("x").X,
            gurobi_model.getVarByName("y").X,
            gurobi_model.getVarByName("z").X,
        ]
    )

    assert np.allclose(pyomoAnswers, gurobiAnswers)


# TODO: Now that Ive made an interface with guroipy and I know that MIPStart values will be initialized, Make a new "Mygurobi" solver that uses this interface. Alternatively, check if "gurobipy_persistent" utilizes MIPStart values.
