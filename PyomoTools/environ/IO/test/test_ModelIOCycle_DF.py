from ..ModelToDF import ModelToDF
from ..LoadModelSolutionFromDF import LoadModelSolutionFromDF
from ...AssertPyomoModelsEqual import AssertPyomoModelsEqual

import pyomo.environ as pyo
import numpy as np
from copy import deepcopy


def test_ModelIOCycle_DF():
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

    model.Obj = pyo.Objective(
        expr=model.V0
        + model.V4
        + sum(
            model.V1[ii]
            + sum(
                model.V2[ii, jj] + sum(model.V3[ii, jj, kk] for kk in s3) for jj in s2
            )
            for ii in s1
        ),
        sense=pyo.maximize,
    )

    modelCopy = deepcopy(model)

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

    assert modelCopy.V0.value is None

    df = ModelToDF(model)

    LoadModelSolutionFromDF(modelCopy, df)

    assert AssertPyomoModelsEqual(model, modelCopy)
