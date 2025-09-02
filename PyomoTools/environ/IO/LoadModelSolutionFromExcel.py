import pyomo.environ as pyo
import pandas as pd

from .LoadModelSolutionFromDF import LoadModelSolutionFromDF


def LoadModelSolutionFromExcel(model: pyo.ConcreteModel, excelFileName: str):
    """
    A function that loads data from an excel workbook into a pyomo model of corresponding structure

    Parameters
    ----------
    model: pyo.ConcreteModel
        The pyomo model you'd like to load the result into
    excelFileName: str
        The name of the excel file you'd like to generate
    """
    allVarData = {}

    # First, load all non-indexed variables
    df = pd.read_excel(excelFileName, sheet_name="NonIndexedVars")
    varNames = df["Variable"].to_numpy()
    varVals = df["Value"].to_numpy()
    for i in range(len(varNames)):
        allVarData[varNames[i]] = varVals[i]

    # Now handle all the indexed vars
    varNames = list(pd.ExcelFile(excelFileName).sheet_names)
    varNames.remove("NonIndexedVars")

    for varName in varNames:
        df = pd.read_excel(excelFileName, sheet_name=varName)
        if "Unnamed: 0" in df.columns:
            df.set_index("Unnamed: 0", inplace=True)
            df.index.name = None
        allVarData[varName] = df

    LoadModelSolutionFromDF(model, allVarData)
