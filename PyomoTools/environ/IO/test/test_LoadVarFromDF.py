from ..LoadVarSolutionFromDF import LoadVarSolutionFromDF

import pyomo.environ as pyo
import numpy as np
import pandas as pd


def test_NonIndexedVar():
    model = pyo.ConcreteModel()

    df = 1.1

    model.X = pyo.Var()
    LoadVarSolutionFromDF(model.X, df)

    assert model.X.value == df


def test_1D_IndexedVar():
    model = pyo.ConcreteModel()

    numVars = 5
    s = list(range(numVars))
    v = np.random.uniform(0, 1, numVars)

    model.MySet = pyo.Set(initialize=s)
    model.X = pyo.Var(model.MySet)

    df = pd.DataFrame(data={"Index": s, "Value": v})

    LoadVarSolutionFromDF(model.X, df)

    newVals = np.zeros(numVars)
    for i in model.MySet:
        newVals[i] = model.X[i].value

    assert np.allclose(newVals, v)


def test_2D_IndexedVar():
    model = pyo.ConcreteModel()

    numVars = 5
    s1 = list(range(numVars))
    s2 = ["A", "B", "C"]

    vals = np.random.uniform(0, 1, (len(s1), len(s2)))

    df = pd.DataFrame(data=vals.T, columns=s1, index=s2)

    model.MySet1 = pyo.Set(initialize=s1)
    model.MySet2 = pyo.Set(initialize=s2)

    model.X = pyo.Var(model.MySet1 * model.MySet2)

    LoadVarSolutionFromDF(model.X, df)

    newVals = np.zeros((len(s1), len(s2)))
    for i, ii in enumerate(s1):
        for j, jj in enumerate(s2):
            newVals[i, j] = model.X[ii, jj].value

    assert np.allclose(vals, newVals)


def test_2D_IndexedVar_MissingData():
    model = pyo.ConcreteModel()

    combos = [(0, "A"), (1, "B"), (2, "C"), (1, "C")]

    model.X = pyo.Var(combos)

    vals = np.random.uniform(0, 1, 4)
    data = [[pd.NA for jj in ["A", "B", "C"]] for ii in [0, 1, 2]]
    data[0][0] = vals[0]
    data[1][1] = vals[1]
    data[2][2] = vals[2]
    data[2][1] = vals[3]

    df = pd.DataFrame(data=data, columns=[0, 1, 2], index=["A", "B", "C"])

    LoadVarSolutionFromDF(model.X, df)

    newVals = np.zeros(4)
    for i, c in enumerate(combos):
        newVals[i] = model.X[c].value

    assert np.allclose(vals, newVals)


def test_3D_IndexedVar():
    model = pyo.ConcreteModel()

    s1 = list(range(5))
    s2 = ["A", "B", "C"]
    s3 = ["!", "@", "#"]

    model.MySet1 = pyo.Set(initialize=s1)
    model.MySet2 = pyo.Set(initialize=s2)
    model.MySet3 = pyo.Set(initialize=s3)
    vals = np.random.uniform(0, 1, (len(s1), len(s2), len(s3)))

    model.X = pyo.Var(model.MySet1 * model.MySet2 * model.MySet3)

    lTotal = len(s1) * len(s2) * len(s3)
    S1 = [None for _ in range(lTotal)]
    S2 = [None for _ in range(lTotal)]
    S3 = [None for _ in range(lTotal)]
    flatVals = [None for _ in range(lTotal)]

    m = 0
    for i, ii in enumerate(s1):
        for j, jj in enumerate(s2):
            for k, kk in enumerate(s3):
                S1[m] = ii
                S2[m] = jj
                S3[m] = kk
                flatVals[m] = vals[i, j, k]
                m += 1

    df = pd.DataFrame(
        data={"Index_1": S1, "Index_2": S2, "Index_3": S3, "Value": flatVals}
    )

    LoadVarSolutionFromDF(model.X, df)

    newFlatVals = np.zeros(lTotal)
    for m in range(lTotal):
        idx = (S1[m], S2[m], S3[m])
        newFlatVals[m] = model.X[idx].value
    assert np.allclose(flatVals, newFlatVals)
