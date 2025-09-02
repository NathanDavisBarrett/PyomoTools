import pyomo.environ as pyo
import pandas as pd


def LoadVarSolutionFromDF(var: pyo.Var, df: pd.DataFrame):
    """
    A function to load values from a pandas dataframe to a pyomo variable.

    **NOTE**: The formatting of the dataframe will be inferred using the same organizational scheme as VarToDF. Please reference the VarToDF documentation for details.

    Parameters
    ----------
    var: pyo.Var
        The pyomo variable you'd like to load data into
    df: pandas dataframe
        The dataframe containing the data you'd like to load
    """
    isIndexed = "Indexed" in str(type(var))

    if isIndexed:
        idxSet = [idx for idx in var.index_set()]
        multiDim = isinstance(idxSet[0], tuple)
        if multiDim:
            if len(idxSet[0]) == 2:
                I1Set = list(df.columns)
                I2Set = list(df.index)
                NAs = df.isna().to_numpy()
                data = df.to_numpy()
                for i, ii in enumerate(I1Set):
                    for j, jj in enumerate(I2Set):
                        if not NAs[j, i]:
                            var[ii, jj].value = data[j, i]
            else:
                idxDim = len(idxSet[0])
                idxNames = [f"Index_{i+1}" for i in range(idxDim)]
                idxArrays = [df[idxNames[i]].to_numpy() for i in range(idxDim)]
                valArray = df["Value"].to_numpy()
                for i in range(len(valArray)):
                    ii = tuple(idxArrays[j][i] for j in range(idxDim))
                    var[ii].value = valArray[i]
        else:
            iis = df["Index"].to_numpy()
            vals = df["Value"].to_numpy()
            for i in range(len(vals)):
                var[iis[i]].value = vals[i]
    else:
        var.value = df
