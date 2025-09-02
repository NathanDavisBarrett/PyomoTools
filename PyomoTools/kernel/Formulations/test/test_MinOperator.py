import pyomo.kernel as pmo
import numpy as np
from ..MinOperator import MinOperator
from ....base.Solvers import DefaultSolver


def Base(includeBinary, fullModel):
    solver = DefaultSolver("MILP")
    model = pmo.block()

    bBounds = (2, 10)
    cBounds = (-50, 100)

    model.A = pmo.variable()
    model.B = pmo.variable(lb=bBounds[0], ub=bBounds[1])
    model.C = pmo.variable(lb=cBounds[0], ub=cBounds[1])
    if includeBinary:
        model.Y = pmo.variable(domain=pmo.Binary)

    model.MO = MinOperator(
        A=model.A,
        B=model.B,
        C=model.C,
        bBounds=bBounds,
        cBounds=cBounds,
        Y=model.Y if includeBinary else None,
        allowMinimizationPotential=fullModel,
    )

    if fullModel:
        model.obj = pmo.objective(expr=model.A, sense=pmo.minimize)

        solver.solve(model)

        assert np.allclose([pmo.value(model.A), pmo.value(model.C)], [-50, -50])

        model.obj.deactivate()

    model.obj1 = pmo.objective(expr=model.A, sense=pmo.maximize)
    solver.solve(model)

    assert np.allclose([pmo.value(model.A), pmo.value(model.B)], [10, 10])

    if not fullModel:
        assert not hasattr(model.MO, "Y")


def test_FullModel_ManualBinary():
    Base(includeBinary=True, fullModel=True)


def test_FullModel_AutoBinary():
    Base(includeBinary=False, fullModel=True)


def test_ConvexModel():
    Base(includeBinary=False, fullModel=False)
