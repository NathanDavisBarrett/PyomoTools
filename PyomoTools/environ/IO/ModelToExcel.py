import pyomo.environ as pyo
import pandas as pd

from .ModelToDF import ModelToDF


def ModelToExcel(model: pyo.ConcreteModel, excelFileName: str):
    """
    A function that converts a pyomo model containing a result to an excel file displaying that result.

    Parameters
    ----------
    model: pyo.ConcreteModel
        The pyomo model containing the results you'd like to convert.
    excelFileName: str
        The name of the excel file you'd like to generate
    """
    vars = ModelToDF(model)

    # Collect all non-indexed variables into one sheet
    nonIndexedVariables = {}

    with pd.ExcelWriter(excelFileName, engine="xlsxwriter") as writer:
        workbook = writer.book

        # First, iterate over each variable.
        for varName in vars:
            if vars[varName] is None:
                continue

            isIndexed = not (isinstance(vars[varName], float) or vars[varName] is None)

            if isIndexed:
                df = vars[varName]
                index = True
                for col in df.columns:
                    if str(col).startswith("Index"):
                        index = False
                        break
                df.to_excel(writer, sheet_name=varName, index=index)

            else:
                nonIndexedVariables[varName] = vars[varName]

        # Now make a new sheet for the non-indexed vars
        if len(nonIndexedVariables) != 0:
            worksheet = workbook.add_worksheet("NonIndexedVars")
            writer.sheets["NonIndexedVars"] = worksheet

            varNames = [v for v in nonIndexedVariables]
            varVals = [nonIndexedVariables[v] for v in varNames]
            data = {"Variable": varNames, "Value": varVals}
            df = pd.DataFrame(data=data)
            df.to_excel(writer, sheet_name="NonIndexedVars", index=False)
