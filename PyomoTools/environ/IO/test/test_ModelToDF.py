import pyomo.environ as pyo
from ..ModelToDF import ModelToDF
import pandas as pd
import numpy as np


def test_ModelToDF():
    model = pyo.ConcreteModel()

    s1 = ["A", "B", "C"]
    s2 = [0, 1, 2, 3]
    s3 = ["!", "@"]

    model.Set1 = pyo.Set(initialize=s1)
    model.Set2 = pyo.Set(initialize=s2)
    model.Set3 = pyo.Set(initialize=s3)

    model.V0 = pyo.Var()
    model.V1 = pyo.Var(model.Set1)
    model.V2 = pyo.Var(model.Set1 * model.Set2)
    model.V3 = pyo.Var(model.Set1 * model.Set2 * model.Set3)
    model.V4 = pyo.Var()

    val0 = np.random.uniform(0, 1)
    val1 = np.random.uniform(0, 1, (len(s1),))
    val2 = np.random.uniform(0, 1, (len(s1), len(s2)))
    val3 = np.random.uniform(0, 1, (len(s1), len(s2), len(s3)))
    val4 = np.random.uniform(0, 1)

    model.V0.value = val0
    model.V4.value = val4
    for i in range(len(s1)):
        ii = s1[i]
        model.V1[ii].value = val1[i]
        for j in range(len(s2)):
            jj = s2[j]
            model.V2[ii, jj].value = val2[i, j]
            for k in range(len(s3)):
                kk = s3[k]
                model.V3[ii, jj, kk].value = val3[i, j, k]

    result = ModelToDF(model)

    assert isinstance(result, dict)
    for i in range(5):
        vName = f"V{i}"
        assert vName in result

    assert isinstance(result["V0"], float)
    assert np.allclose(
        [
            result["V0"],
        ],
        [val0],
    )

    assert isinstance(result["V1"], pd.DataFrame)
    cols = result["V1"].columns
    assert "Index" in cols
    assert "Value" in cols
    indices = result["V1"]["Index"].to_numpy()
    dfVals = result["V1"]["Value"].to_numpy()
    newVals = np.zeros(len(s1))
    for m in range(len(indices)):
        ii = indices[m]
        val = dfVals[m]
        i = s1.index(ii)
        newVals[i] = val
    assert np.allclose(val1, newVals)

    assert isinstance(result["V2"], pd.DataFrame)
    for i in model.Set1:
        assert i in result["V2"].columns
    for j in model.Set2:
        assert j in result["V2"].index
    newVals = np.zeros((len(s1), len(s2)))
    for i, ii in enumerate(s1):
        for j, jj in enumerate(s2):
            newVals[i, j] = result["V2"].loc[jj][ii]
    assert np.allclose(val2, newVals)

    assert isinstance(result["V3"], pd.DataFrame)
    for m in range(3):
        assert f"Index_{m+1}" in result["V3"].columns
    newVals = np.zeros((len(s1), len(s2), len(s3)))
    I1s = result["V3"]["Index_1"].to_numpy()
    I2s = result["V3"]["Index_2"].to_numpy()
    I3s = result["V3"]["Index_3"].to_numpy()
    dfVals = result["V3"]["Value"].to_numpy()
    for m in range(len(I1s)):
        ii = I1s[m]
        jj = I2s[m]
        kk = I3s[m]
        val = dfVals[m]

        i = s1.index(ii)
        j = s2.index(jj)
        k = s3.index(kk)

        newVals[i, j, k] = val
    assert np.allclose(val3, newVals)

    assert isinstance(result["V4"], float)
    assert np.allclose(
        [
            result["V4"],
        ],
        [val4],
    )
