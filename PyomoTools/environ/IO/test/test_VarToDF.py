from ..VarToDF import VarToDF

import pyomo.environ as pyo
import numpy as np
import pandas as pd


def test_NonIndexedVar():
    model = pyo.ConcreteModel()

    testVal = 1.1

    model.X = pyo.Var()
    model.X.value = testVal

    result = VarToDF(model.X)

    assert isinstance(result, float)
    assert np.allclose(
        [
            result,
        ],
        [
            testVal,
        ],
    )


def test_1D_IndexedVar():
    model = pyo.ConcreteModel()

    numVars = 5
    model.MySet = pyo.Set(initialize=list(range(numVars)))
    vals = np.random.uniform(0, 1, numVars)

    model.X = pyo.Var(model.MySet)
    for i in model.MySet:
        model.X[i].value = vals[i]

    result = VarToDF(model.X)

    assert isinstance(result, pd.DataFrame)
    cols = result.columns
    assert "Index" in cols
    assert "Value" in cols

    indices = result["Index"].to_numpy()
    dfVals = result["Value"].to_numpy()

    newVals = np.zeros(numVars)
    newVals[indices] = dfVals

    assert np.allclose(vals, newVals)


def test_1D_IndexedVar_2():
    model = pyo.ConcreteModel()

    numVars = 5
    model.MySet = pyo.Set(initialize=[(i,) for i in range(numVars)])
    vals = np.random.uniform(0, 1, numVars)

    model.X = pyo.Var(model.MySet)
    for i in model.MySet:
        model.X[i].value = vals[i]

    result = VarToDF(model.X)

    assert isinstance(result, pd.DataFrame)
    cols = result.columns
    assert "Index" in cols
    assert "Value" in cols

    indices = result["Index"].to_numpy()
    dfVals = result["Value"].to_numpy()

    newVals = np.zeros(numVars)
    newVals[indices] = dfVals

    assert np.allclose(vals, newVals)


def test_2D_IndexedVar():
    model = pyo.ConcreteModel()

    numVars = 5
    model.MySet1 = pyo.Set(initialize=list(range(numVars)))
    model.MySet2 = pyo.Set(initialize=["A", "B", "C"])
    vals = np.random.uniform(0, 1, (numVars, 3))

    model.X = pyo.Var(model.MySet1 * model.MySet2)
    for i in model.MySet1:
        for j in model.MySet2:
            jj = ["A", "B", "C"].index(j)
            model.X[i, j].value = vals[i, jj]

    result = VarToDF(model.X)

    assert isinstance(result, pd.DataFrame)

    for i in model.MySet1:
        assert i in result.columns

    for j in model.MySet2:
        assert j in result.index

    newVals = np.zeros((numVars, 3))
    for i in model.MySet1:
        for j in model.MySet2:
            jj = ["A", "B", "C"].index(j)
            newVals[i, jj] = result.loc[j][i]

    assert np.allclose(vals, newVals)


def test_2D_IndexedVar_MissingData():
    model = pyo.ConcreteModel()

    combos = [(0, "A"), (1, "B"), (2, "C")]
    vals = np.random.uniform(0, 1, 3)

    model.X = pyo.Var(combos)
    for i, idx in enumerate(combos):
        model.X[idx].value = vals[i]

    result = VarToDF(model.X)

    assert isinstance(result, pd.DataFrame)

    for i in [0, 1, 2]:
        assert i in result.columns

    for j in ["A", "B", "C"]:
        assert j in result.index

    newVals = np.zeros(3)
    for i in [0, 1, 2]:
        for j in ["A", "B", "C"]:
            val = result.loc[j][i]
            if (i, j) not in combos:
                assert pd.isna(val)
            else:
                newVals[i] = val

    assert np.allclose(vals, newVals)


def test_3D_IndexedVar():
    model = pyo.ConcreteModel()

    numVars = 5
    model.MySet1 = pyo.Set(initialize=list(range(numVars)))
    model.MySet2 = pyo.Set(initialize=["A", "B", "C"])
    model.MySet3 = pyo.Set(initialize=["!", "@", "#"])
    vals = np.random.uniform(0, 1, (numVars, 3, 3))

    model.X = pyo.Var(model.MySet1 * model.MySet2 * model.MySet3)
    for i in model.MySet1:
        for j in model.MySet2:
            jj = ["A", "B", "C"].index(j)
            for k in model.MySet3:
                kk = ["!", "@", "#"].index(k)
                model.X[i, j, k].value = vals[i, jj, kk]

    result = VarToDF(model.X)

    assert isinstance(result, pd.DataFrame)
    assert "Index_1" in result.columns
    assert "Index_2" in result.columns
    assert "Index_3" in result.columns
    assert "Index_4" not in result.columns

    newVals = np.zeros((numVars, 3, 3))

    I1s = result["Index_1"].to_numpy()
    I2s = result["Index_2"].to_numpy()
    I3s = result["Index_3"].to_numpy()
    dfVals = result["Value"].to_numpy()

    for i in range(len(I1s)):
        i = I1s[i]
        j = I2s[i]
        k = I3s[i]
        val = dfVals[i]

        jj = ["A", "B", "C"].index(j)
        kk = ["!", "@", "#"].index(k)

        newVals[i, jj, kk] = val

    assert np.allclose(vals, newVals)
