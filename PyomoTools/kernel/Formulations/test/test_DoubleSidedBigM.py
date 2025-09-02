import pyomo.kernel as pmo

from ..DoubleSidedBigM import DoubleSidedBigM
from ....base.Solvers import DefaultSolver


def test_Construction():
    xBounds = [-2, 10]

    model = pmo.block()
    model.Y = pmo.variable(domain=pmo.Reals)
    model.X = pmo.variable(domain=pmo.Reals)
    model.Z = pmo.variable(domain=pmo.Binary)

    model.DSBM = DoubleSidedBigM(
        A=model.Y, B=model.X, X=model.Z, Bmin=xBounds[0], Bmax=xBounds[1]
    )
    # model.DSBM.Plot()


def test_ManualBinary_NoC():
    xBounds = [-2, 10]

    model = pmo.block()
    model.Y = pmo.variable(domain=pmo.Reals)
    model.X = pmo.variable(domain=pmo.Reals)
    model.Z = pmo.variable(domain=pmo.Binary)

    model.DSBM = DoubleSidedBigM(
        A=model.Y, B=model.X, X=model.Z, Bmin=xBounds[0], Bmax=xBounds[1]
    )

    model.obj = pmo.objective(model.Y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    assert model.X.value == xBounds[1]
    assert model.Y.value == xBounds[1]
    assert model.Z.value == 1

    model.obj.deactivate()
    model.obj1 = pmo.objective(model.Y, sense=pmo.minimize)

    solver.solve(model)
    assert model.X.value == xBounds[0]
    assert model.Y.value == xBounds[0]
    assert model.Z.value == 1

    model.Constr = pmo.constraint(model.Z == 0)
    solver.solve(model)

    assert pmo.value(model.Y) == 0
    assert pmo.value(model.Z) == 0

    model.obj1.deactivate()
    model.obj.activate()
    solver.solve(model)

    assert pmo.value(model.Y) == 0
    assert pmo.value(model.Z) == 0


def test_NonIndexed_ManualBinary_YesC():
    xBounds = [-2, 10]

    model = pmo.block()
    model.Y = pmo.variable(domain=pmo.Reals)
    model.X = pmo.variable(domain=pmo.Reals)
    model.Z = pmo.variable(domain=pmo.Binary)
    model.C = pmo.variable(domain=pmo.Reals, lb=0, ub=1)

    model.DSBM = DoubleSidedBigM(
        A=model.Y, B=model.X, X=model.Z, C=model.C, Bmin=xBounds[0], Bmax=xBounds[1]
    )

    model.obj = pmo.objective(model.Y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    assert model.C.value == 1
    assert model.X.value == xBounds[1]
    assert model.Y.value == xBounds[1] + 1
    assert model.Z.value == 1

    model.obj.deactivate()
    model.obj1 = pmo.objective(model.Y, sense=pmo.minimize)

    solver.solve(model)
    assert model.X.value == xBounds[0]
    assert model.Y.value == xBounds[0]
    assert model.Z.value == 1
    assert model.C.value == 0

    model.Constr = pmo.constraint(model.Z == 0)
    solver.solve(model)

    assert pmo.value(model.Y) == 0
    assert pmo.value(model.Z) == 0
    assert pmo.value(model.C) == 0

    model.obj1.deactivate()
    model.obj.activate()
    solver.solve(model)

    assert pmo.value(model.Y) == 1
    assert pmo.value(model.C) == 1
    assert pmo.value(model.Z) == 0


def test_NonIndexed_AutoBinary():
    xBounds = [-2, 10]

    model = pmo.block()
    model.Y = pmo.variable(domain=pmo.Reals)
    model.X = pmo.variable(domain=pmo.Reals)

    model.DSBM = DoubleSidedBigM(A=model.Y, B=model.X, Bmin=xBounds[0], Bmax=xBounds[1])

    model.obj = pmo.objective(model.Y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    assert model.X.value == xBounds[1]
    assert model.Y.value == xBounds[1]

    model.obj.deactivate()
    model.obj1 = pmo.objective(model.Y, sense=pmo.minimize)

    solver.solve(model)
    assert model.X.value == xBounds[0]
    assert model.Y.value == xBounds[0]
