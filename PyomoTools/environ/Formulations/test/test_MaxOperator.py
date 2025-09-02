import pyomo.environ as pyo
import numpy as np
from ..MaxOperator import MaxOperator
from ....base.Solvers import DefaultSolver


def nonIndexed_Base(includeBinary, fullModel):
    solver = DefaultSolver("MILP")
    model = pyo.ConcreteModel()

    bBounds = (2, 10)
    cBounds = (-50, 100)

    model.A = pyo.Var()
    model.B = pyo.Var(bounds=bBounds)
    model.C = pyo.Var(bounds=cBounds)
    if includeBinary:
        model.Y = pyo.Var(domain=pyo.Binary)

    relationshipBaseName = "MaxOperatorTest"

    MaxOperator(
        model=model,
        A=model.A,
        B=model.B,
        C=model.C,
        bBounds=bBounds,
        cBounds=cBounds,
        Y=model.Y if includeBinary else None,
        allowMaximizationPotential=fullModel,
        relationshipBaseName=relationshipBaseName,
    )

    if fullModel:
        model.obj = pyo.Objective(expr=model.A, sense=pyo.maximize)

        solver.solve(model)

        assert np.allclose([pyo.value(model.A), pyo.value(model.C)], [100, 100])

        model.obj.deactivate()

    model.obj1 = pyo.Objective(expr=model.A, sense=pyo.minimize)
    solver.solve(model)

    assert np.allclose([pyo.value(model.A), pyo.value(model.B)], [2, 2])

    if not fullModel:
        assert not hasattr(model, f"{relationshipBaseName}_Y")


def test_NonIndexed_FullModel_ManualBinary():
    nonIndexed_Base(includeBinary=True, fullModel=True)


def test_NonIndexed_FullModel_AutoBinary():
    nonIndexed_Base(includeBinary=False, fullModel=True)


def test_NonIndexed_ConvexModel():
    nonIndexed_Base(includeBinary=False, fullModel=False)


def indexed_Base(includeBinary, fullModel):
    solver = DefaultSolver("MILP")
    model = pyo.ConcreteModel()

    n = 3
    model.TestSet = pyo.Set(initialize=list(range(n)))

    bBounds = (2, 10)
    cBounds = (-50, 100)

    model.A = pyo.Var(model.TestSet * model.TestSet)
    model.B = pyo.Var(model.TestSet * model.TestSet, bounds=bBounds)
    model.C = pyo.Var(model.TestSet * model.TestSet, bounds=cBounds)
    if includeBinary:
        model.Y = pyo.Var(model.TestSet * model.TestSet, domain=pyo.Binary)

    relationshipBaseName = "MaxOperatorTest"

    MaxOperator(
        model=model,
        A=model.A,
        B=model.B,
        C=model.C,
        bBounds={ii: bBounds for ii in model.TestSet * model.TestSet},
        cBounds={ii: cBounds for ii in model.TestSet * model.TestSet},
        Y=model.Y if includeBinary else None,
        allowMaximizationPotential=fullModel,
        itrSet=model.TestSet * model.TestSet,
        relationshipBaseName=relationshipBaseName,
    )

    if fullModel:
        model.obj = pyo.Objective(
            expr=sum(model.A[i1, i2] for i1, i2 in model.TestSet * model.TestSet),
            sense=pyo.maximize,
        )

        solver.solve(model)

        for ii in model.TestSet * model.TestSet:
            assert np.allclose(
                [pyo.value(model.A[ii]), pyo.value(model.C[ii])], [100, 100]
            )

        model.obj.deactivate()

    model.obj1 = pyo.Objective(
        expr=sum(model.A[i1, i2] for i1, i2 in model.TestSet * model.TestSet),
        sense=pyo.minimize,
    )
    solver.solve(model)

    for ii in model.TestSet * model.TestSet:
        assert np.allclose([pyo.value(model.A[ii]), pyo.value(model.B[ii])], [2, 2])

    if not fullModel:
        assert not hasattr(model, f"{relationshipBaseName}_Y")


def test_Indexed_FullModel_ManualBinary():
    indexed_Base(includeBinary=True, fullModel=True)


def test_Indexed_FullModel_AutoBinary():
    indexed_Base(includeBinary=False, fullModel=True)


def test_Indexed_ConvexModel():
    indexed_Base(includeBinary=False, fullModel=False)
