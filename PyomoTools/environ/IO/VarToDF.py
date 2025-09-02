import pyomo.environ as pyo
import pandas as pd
from typing import Union


def VarToDF(var: pyo.Var) -> Union[pd.DataFrame, float]:
    """
    A function to convert a pyomo variable (containing a solution) to a pandas dataframe (if the variable is indexed) or a float (if the variable is not indexed).

    **NOTE**: The format of the outputted dataframe depends on the dimensionality of the variable's index:

    * If the variable is not indexed, it's value will simply by a float
    * If the dimensionality of the variable index is 1, the resulting dataframe will contain two columns: one called "Index" values with each index value and one called "Value" containing the corresponding value
    * If the dimensionality of the variable is 2, the resulting dataframe will contain as many columns as there are elements in the first index set and as many rows as there are elements in the second index set. Each element will therefore correspond with a row index and a column index. Row/column combinations that are not included in this variable's index will be left blank.
    * For all other dimensionalities, a number of columns corresponding to the dimensionality of the index will be labeled (e.g. "Index #1", "Index #2", etc.) following these columns will be a "Value" column containing the value of the variable at that index.

    Parameters
    ----------
    var: pyo.Var
        The pyomo variable you'd like to convert

    Returns
    -------
    result: pandas dataframe or float containing the value of the variable provided
    """
    isIndexed = "Indexed" in str(type(var))

    if isIndexed:
        idxSet = [idx for idx in var.index_set()]
        if len(idxSet) == 0:
            return None
        multiDim = isinstance(idxSet[0], tuple)
        if multiDim:
            if len(idxSet[0]) == 2:
                idxSet1 = []
                idxSet2 = []
                idx1Map = {}
                idx2Map = {}
                for i1, i2 in idxSet:
                    if i1 not in idxSet1:
                        idx1Map[i1] = len(idxSet1)
                        idxSet1.append(i1)
                    if i2 not in idxSet2:
                        idx2Map[i2] = len(idxSet2)
                        idxSet2.append(i2)

                data = [
                    [pd.NA for _ in range(len(idxSet1))] for _ in range(len(idxSet2))
                ]
                for i1, i2 in idxSet:
                    i = idx2Map[i2]
                    j = idx1Map[i1]
                    val = pyo.value(var[i1, i2], exception=False)
                    data[i][j] = val

                df = pd.DataFrame(data=data, columns=idxSet1, index=idxSet2)
            else:
                idxDim = len(idxSet[0])
                idxArrays = [
                    [idxSet[j][i] for j in range(len(idxSet))] for i in range(idxDim)
                ]
                idxNames = [f"Index_{i+1}" for i in range(idxDim)]
                idxTypes = [type(idxSet[0][i]) for i in range(idxDim)]

                values = [pyo.value(var[idx], exception=False) for idx in idxSet]

                data = {idxNames[i]: idxArrays[i] for i in range(idxDim)}
                data["Value"] = values

                df = pd.DataFrame(data=data)
                for i in range(idxDim):
                    idxName = idxNames[i]
                    df[idxName] = df[idxName].astype(idxTypes[i])
        else:
            idxDim = 1
            idxArrays = [
                [idxSet[j] for j in range(len(idxSet))],
            ]
            idxNames = [
                "Index",
            ]

            idxType = type(idxSet[0])

            values = [pyo.value(var[idx], exception=False) for idx in idxSet]

            data = {idxNames[i]: idxArrays[i] for i in range(idxDim)}
            data["Value"] = values

            df = pd.DataFrame(data=data)
            df["Index"] = df["Index"].astype(idxType)

        return df
    else:
        return pyo.value(var, exception=False)
