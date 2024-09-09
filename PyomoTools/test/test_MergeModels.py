import pyomo.environ as pyo
import numpy as np
import inspect

from ..MergeModels import MergeModels
from ..Solvers import DefaultSolver

def test_CarryBounds():
    model1 = pyo.ConcreteModel()
    model1.X = pyo.Var(bounds=(-10,10))
    model1.Y = pyo.Var()
    model1.C1 = pyo.Constraint(expr=model1.Y == pyo.exp(model1.X))

    model2 = pyo.ConcreteModel()
    model2.Y = pyo.Var()
    model2.Z = pyo.Var()
    model2.C1 = pyo.Constraint(expr=model2.Z == -2 * model2.Y)

    model3 = pyo.ConcreteModel()
    MergeModels(model3,{"model1":model1, "model2": model2})

    model3.C1 = pyo.Constraint(expr=model3.model1.Y == model3.model2.Y)

    model3.Obj = pyo.Objective(expr=model3.model2.Z,sense = pyo.maximize)

    solver = DefaultSolver(problemType="NLP")
    solver.solve(model3)

    result = np.array([pyo.value(model3.model1.X),pyo.value(model3.model1.Y),pyo.value(model3.model2.Y),pyo.value(model3.model2.Z)])
    expected = np.array([-10,np.exp(-10),np.exp(-10),-2*np.exp(-10)])

    assert np.allclose(result,expected)
    
def test_IndexedVars():
    model1 = pyo.ConcreteModel()
    model1.Set1 = pyo.Set(initialize=[1,2,3])
    model1.X = pyo.Var(model1.Set1,bounds=(-10,10))
    model1.Y = pyo.Var(model1.Set1)
    model1.C1 = pyo.Constraint(model1.Set1,rule=lambda _,i: model1.Y[i] == i * pyo.exp(model1.X[i]))
    model1.Z = pyo.Var()
    model1.C2 = pyo.Constraint(expr=sum(model1.Y[i] for i in model1.Set1) == model1.Z)

    model2 = pyo.ConcreteModel()
    model2.Set1 = pyo.Set(initialize=["A","B","C"])
    model2.Y = pyo.Var(model2.Set1,bounds=(-1,1))
    model2.Z = pyo.Var(model2.Set1)
    model2.C1 = pyo.Constraint(model2.Set1, rule=lambda _,i:model2.Z[i] == -model2.Y[i])
    model2.A = pyo.Var()
    model2.C2 = pyo.Constraint(expr=sum(model2.Y[i] for i in model2.Set1) == model2.A)


    model3 = pyo.ConcreteModel()
    MergeModels(model3,{"model1":model1, "model2": model2})

    model3.B = pyo.Var()
    model3.C1 = pyo.Constraint(expr= model3.B == model3.model1.Z + model3.model2.A)

    model3.Obj = pyo.Objective(expr=model3.B,sense = pyo.maximize)

    solver = DefaultSolver(problemType="NLP")
    solver.solve(model3)

    #First, assert model1 solution
    X1_result = np.array([pyo.value(model3.model1.X[i]) for i in model3.model1.Set1])
    expected = np.ones(len(model3.model1.Set1)) * 10
    assert np.allclose(X1_result,expected)

    Y1_result = np.array([pyo.value(model3.model1.Y[i]) for i in model3.model1.Set1])
    expected = np.array([i * np.exp(10) for i in model3.model1.Set1])
    assert np.allclose(Y1_result,expected)

    Y2_result = np.array([pyo.value(model3.model2.Y[i]) for i in model3.model2.Set1])
    expected = np.ones(len(model3.model2.Set1))
    assert np.allclose(Y2_result,expected)

    Z2_result = np.array([pyo.value(model3.model2.Z[i]) for i in model3.model2.Set1])
    expected = np.ones(len(model3.model2.Set1)) * -1
    assert np.allclose(Z2_result,expected)

def test_CarrySolutions():
    model1 = pyo.ConcreteModel()
    model1.Set1 = pyo.Set(initialize=[1,2,3])
    model1.X = pyo.Var(model1.Set1,bounds=(-10,10))
    model1.Y = pyo.Var(model1.Set1)
    model1.C1 = pyo.Constraint(model1.Set1,rule=lambda _,i: model1.Y[i] == i * pyo.exp(model1.X[i]))

    for i in model1.Set1:
        model1.X[i].value = 10
        yi = i * np.exp(10)
        model1.Y[i].value = yi

    

    model2 = pyo.ConcreteModel()
    model2.Y = pyo.Var(bounds=(-1,1))
    model2.Z = pyo.Var()
    model2.C1 = pyo.Constraint(expr=model2.Z == -model2.Y)
    model2.Y.value = 1
    model2.Z.value = -1


    model3 = pyo.ConcreteModel()
    MergeModels(model3,{"model1":model1, "model2": model2})

    model3.Obj = pyo.Objective(expr=sum(model3.model1.Y[i] for i in model3.model1.Set1)+ model3.model2.Y,sense = pyo.maximize)

    objVal = pyo.value(model3.Obj)
    expected = sum(i * np.exp(10) for i in model1.Set1) + 1
    assert np.allclose([objVal,],[expected,])

def test_CarryActivations():
    model1 = pyo.ConcreteModel()
    model1.Set1 = pyo.Set(initialize=[1,2,3])
    model1.X = pyo.Var(model1.Set1,bounds=(-10,10))
    model1.Y = pyo.Var(model1.Set1)
    model1.C1 = pyo.Constraint(model1.Set1,rule=lambda _,i: model1.Y[i] == i * pyo.exp(model1.X[i]))
    model1.C1.deactivate()
    model1.C2 = pyo.Constraint(model1.Set1,rule=lambda _,i: model1.Y[i] == i * model1.X[i])
    model1.Z = pyo.Var()
    model1.C3 = pyo.Constraint(expr= sum(model1.Y[i] for i in model1.Set1) == model1.Z)

    

    model2 = pyo.ConcreteModel()
    model2.Y = pyo.Var(bounds=(-1,1))
    model2.Z = pyo.Var()
    model2.C1 = pyo.Constraint(expr=model2.Z == -model2.Y)
    model2.C1.deactivate()
    model2.C2 = pyo.Constraint(expr=model2.Z == model2.Y)


    model3 = pyo.ConcreteModel()
    MergeModels(model3,{"model1":model1, "model2": model2})

    model3.Obj = pyo.Objective(expr=model3.model1.Z + model3.model2.Z,sense = pyo.maximize)

    solver = DefaultSolver("NLP")
    solver.solve(model3)

    result = np.array([
        *[pyo.value(model3.model1.X[i]) for i in model1.Set1],
        *[pyo.value(model3.model1.Y[i]) for i in model1.Set1],
        pyo.value(model3.model1.Z),
        pyo.value(model3.model2.Y),
        pyo.value(model3.model2.Z)
    ])
    expected = np.array([
        10,10,10,
        10,20,30,
        60,
        1,
        1
    ])
    assert np.allclose(result,expected)

def test_CarryVariableDomains():
    model1 = pyo.ConcreteModel()
    model1.Set1 = pyo.Set(initialize=[1,2,3])
    model1.X = pyo.Var(model1.Set1,domain=pyo.Integers,bounds=(-10.5,10.5))
    model1.Y = pyo.Var(model1.Set1)
    model1.C1 = pyo.Constraint(model1.Set1,rule=lambda _,i: model1.Y[i] == i * pyo.exp(model1.X[i]))
    model1.C1.deactivate()
    model1.C2 = pyo.Constraint(model1.Set1,rule=lambda _,i: model1.Y[i] == i * model1.X[i])
    model1.Z = pyo.Var()
    model1.C3 = pyo.Constraint(expr= sum(model1.Y[i] for i in model1.Set1) == model1.Z)

    

    model2 = pyo.ConcreteModel()
    model2.Y = pyo.Var(bounds=(-1.5,1.5),domain=pyo.Integers)
    model2.Z = pyo.Var()
    model2.C1 = pyo.Constraint(expr=model2.Z == -model2.Y)
    model2.C1.deactivate()
    model2.C2 = pyo.Constraint(expr=model2.Z == model2.Y)


    model3 = pyo.ConcreteModel()
    MergeModels(model3,{"model1":model1, "model2": model2})

    model3.Obj = pyo.Objective(expr=model3.model1.Z + model3.model2.Z,sense = pyo.maximize)

    solver = DefaultSolver("MINLP")
    solver.solve(model3)

    result = np.array([
        *[pyo.value(model3.model1.X[i]) for i in model1.Set1],
        *[pyo.value(model3.model1.Y[i]) for i in model1.Set1],
        pyo.value(model3.model1.Z),
        pyo.value(model3.model2.Y),
        pyo.value(model3.model2.Z)
    ])
    expected = np.array([
        10,10,10,
        10,20,30,
        60,
        1,
        1
    ])
    assert np.allclose(result,expected)