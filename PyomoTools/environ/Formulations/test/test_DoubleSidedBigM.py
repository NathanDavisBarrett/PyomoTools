import pyomo.environ as pyo
import numpy as np

from ..DoubleSidedBigM import DoubleSidedBigM
from ....base.Solvers import DefaultSolver

def test_NonIndexed_ManualBinary_NoC():
    xBounds = [-2,10]

    model = pyo.ConcreteModel()
    model.Y = pyo.Var(domain=pyo.Reals)
    model.X = pyo.Var(domain=pyo.Reals)
    model.Z = pyo.Var(domain=pyo.Binary)

    DoubleSidedBigM(
        model=model,
        A=model.Y,
        B=model.X,
        X=model.Z,
        Bmin=xBounds[0],
        Bmax=xBounds[1]
    )
    
    model.obj = pyo.Objective(expr=model.Y,sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    assert model.X.value == xBounds[1]
    assert model.Y.value == xBounds[1]
    assert model.Z.value == 1

    model.obj.deactivate()
    model.obj1 = pyo.Objective(expr=model.Y,sense=pyo.minimize)

    solver.solve(model)
    assert model.X.value == xBounds[0]
    assert model.Y.value == xBounds[0]
    assert model.Z.value == 1

    model.Constr = pyo.Constraint(expr=model.Z == 0)
    solver.solve(model)

    assert pyo.value(model.Y) == 0
    assert pyo.value(model.Z) == 0

    model.obj1.deactivate()
    model.obj.activate()
    solver.solve(model)

    assert pyo.value(model.Y) == 0
    assert pyo.value(model.Z) == 0

def test_NonIndexed_ManualBinary_YesC():
    xBounds = [-2,10]

    model = pyo.ConcreteModel()
    model.Y = pyo.Var(domain=pyo.Reals)
    model.X = pyo.Var(domain=pyo.Reals)
    model.Z = pyo.Var(domain=pyo.Binary)
    model.C = pyo.Var(domain=pyo.Reals,bounds=(0,1))

    DoubleSidedBigM(
        model=model,
        A=model.Y,
        B=model.X,
        X=model.Z,
        C=model.C,
        Bmin=xBounds[0],
        Bmax=xBounds[1]
    )
    
    model.obj = pyo.Objective(expr=model.Y,sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    assert model.C.value == 1
    assert model.X.value == xBounds[1]
    assert model.Y.value == xBounds[1] + 1
    assert model.Z.value == 1
    

    model.obj.deactivate()
    model.obj1 = pyo.Objective(expr=model.Y,sense=pyo.minimize)

    solver.solve(model)
    assert model.X.value == xBounds[0]
    assert model.Y.value == xBounds[0]
    assert model.Z.value == 1
    assert model.C.value == 0

    model.Constr = pyo.Constraint(expr=model.Z == 0)
    solver.solve(model)

    assert pyo.value(model.Y) == 0
    assert pyo.value(model.Z) == 0
    assert pyo.value(model.C) == 0

    model.obj1.deactivate()
    model.obj.activate()
    solver.solve(model)

    assert pyo.value(model.Y) == 1
    assert pyo.value(model.C) == 1
    assert pyo.value(model.Z) == 0

def test_NonIndexed_AutoBinary():
    xBounds = [-2,10]

    model = pyo.ConcreteModel()
    model.Y = pyo.Var(domain=pyo.Reals)
    model.X = pyo.Var(domain=pyo.Reals)

    DoubleSidedBigM(
        model=model,
        A=model.Y,
        B=model.X,
        Bmin=xBounds[0],
        Bmax=xBounds[1]
    )
    
    model.obj = pyo.Objective(expr=model.Y,sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    assert model.X.value == xBounds[1]
    assert model.Y.value == xBounds[1]

    model.obj.deactivate()
    model.obj1 = pyo.Objective(expr=model.Y,sense=pyo.minimize)

    solver.solve(model)
    assert model.X.value == xBounds[0]
    assert model.Y.value == xBounds[0]

def test_Indexed_ManualBinary_NoC():
    model = pyo.ConcreteModel()
    model.TestSet = pyo.Set(initialize=["A","B","C"])
    iSet = list(model.TestSet * model.TestSet)
    model.Y = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Reals)
    model.X = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Reals)
    model.Z = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Binary)

    n = len(model.TestSet)**2
    xMin = np.random.uniform(-10,10,n)
    xMax = xMin + np.random.uniform(0,10,n)

    xMin = {iSet[i]: xMin[i] for i in range(n)}
    xMax = {iSet[i]: xMax[i] for i in range(n)}

    DoubleSidedBigM(
        model=model,
        A=model.Y,
        B=model.X,
        X=model.Z,
        Bmin=xMin,
        Bmax=xMax,
        itrSet=iSet
    )
    
    model.obj = pyo.Objective(expr=sum(model.Y[i1,i2] for i1,i2 in model.TestSet * model.TestSet),sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    for ii in iSet:
        if xMax[ii] < 0:
            assert pyo.value(model.Y[ii]) == 0
            assert pyo.value(model.Z[ii]) == 0
        else:
            assert np.allclose([pyo.value(model.Y[ii])], [xMax[ii]])
            assert np.allclose([pyo.value(model.X[ii])], [xMax[ii]])
            assert pyo.value(model.Z[ii]) == 1



    model.obj.deactivate()
    model.obj1 = pyo.Objective(expr=sum(model.Y[i1,i2] for i1,i2 in model.TestSet * model.TestSet),sense=pyo.minimize)

    solver.solve(model)
    for ii in iSet:
        if xMin[ii] > 0:
            assert pyo.value(model.Y[ii]) == 0
            assert pyo.value(model.Z[ii]) == 0
        else:
            assert np.allclose([pyo.value(model.Y[ii])] ,[xMin[ii]])
            assert np.allclose([pyo.value(model.X[ii])] ,[xMin[ii]])
            assert pyo.value(model.Z[ii]) == 1

def test_Indexed_ManualBinary_YesC():
    model = pyo.ConcreteModel()
    model.TestSet = pyo.Set(initialize=["A","B","C"])
    iSet = list(model.TestSet * model.TestSet)
    model.Y = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Reals)
    model.X = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Reals)
    model.Z = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Binary)
    model.C = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Binary)

    n = len(model.TestSet)**2
    xMin = np.random.uniform(-10,10,n)
    xMax = xMin + np.random.uniform(0,10,n)

    xMin = {iSet[i]: xMin[i] for i in range(n)}
    xMax = {iSet[i]: xMax[i] for i in range(n)}

    DoubleSidedBigM(
        model=model,
        A=model.Y,
        B=model.X,
        X=model.Z,
        C=model.C,
        Bmin=xMin,
        Bmax=xMax,
        itrSet=iSet
    )
    
    model.obj = pyo.Objective(expr=sum(model.Y[i1,i2] for i1,i2 in model.TestSet * model.TestSet),sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    for ii in iSet:
        if xMax[ii] < 0:
            assert np.allclose([pyo.value(model.Y[ii]),],[1,])
            assert pyo.value(model.Z[ii]) == 0
            assert pyo.value(model.C[ii]) == 1
        else:
            assert np.allclose([pyo.value(model.Y[ii])], [xMax[ii] + 1])
            assert np.allclose([pyo.value(model.X[ii])], [xMax[ii]])
            assert pyo.value(model.Z[ii]) == 1
            assert pyo.value(model.C[ii]) == 1




    model.obj.deactivate()
    model.obj1 = pyo.Objective(expr=sum(model.Y[i1,i2] for i1,i2 in model.TestSet * model.TestSet),sense=pyo.minimize)

    solver.solve(model)
    for ii in iSet:
        if xMin[ii] > 0:
            assert pyo.value(model.Y[ii]) == 0
            assert pyo.value(model.Z[ii]) == 0
            assert pyo.value(model.C[ii]) == 0
        else:
            assert np.allclose([pyo.value(model.Y[ii])],[xMin[ii]])
            assert np.allclose([pyo.value(model.X[ii])],[xMin[ii]])
            assert pyo.value(model.Z[ii]) == 1
            assert pyo.value(model.C[ii]) == 0

def test_Indexed_AutoBinary():
    model = pyo.ConcreteModel()
    model.TestSet = pyo.Set(initialize=["A","B","C"])
    iSet = list(model.TestSet * model.TestSet)
    model.Y = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Reals)
    model.X = pyo.Var(model.TestSet * model.TestSet,domain=pyo.Reals)

    n = len(model.TestSet)**2
    xMin = np.random.uniform(-10,10,n)
    xMax = xMin + np.random.uniform(0,10,n)

    xMin = {iSet[i]: xMin[i] for i in range(n)}
    xMax = {iSet[i]: xMax[i] for i in range(n)}

    DoubleSidedBigM(
        model=model,
        A=model.Y,
        B=model.X,
        Bmin=xMin,
        Bmax=xMax,
        itrSet=iSet
    )
    
    model.obj = pyo.Objective(expr=sum(model.Y[i1,i2] for i1,i2 in model.TestSet * model.TestSet),sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    solver.solve(model)
    for ii in iSet:
        if xMax[ii] < 0:
            assert pyo.value(model.Y[ii]) == 0
        else:
            assert np.allclose([pyo.value(model.Y[ii])], [xMax[ii]])
            assert np.allclose([pyo.value(model.X[ii])], [xMax[ii]])



    model.obj.deactivate()
    model.obj1 = pyo.Objective(expr=sum(model.Y[i1,i2] for i1,i2 in model.TestSet * model.TestSet),sense=pyo.minimize)

    solver.solve(model)
    for ii in iSet:
        if xMin[ii] > 0:
            assert pyo.value(model.Y[ii]) == 0
        else:
            assert np.allclose([pyo.value(model.Y[ii])], [xMin[ii]])
            assert np.allclose([pyo.value(model.X[ii])], [xMin[ii]])
