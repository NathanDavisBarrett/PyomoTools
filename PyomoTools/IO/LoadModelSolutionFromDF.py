import pyomo.environ as pyo
import re

from .LoadVarSolutionFromDF import LoadVarSolutionFromDF
from ..MergeableModel import MergableModel

def LoadModelSolutionFromDF(model:pyo.ConcreteModel,data:dict):
    """
    A function to load values from a dict of pandas dataframes to a pyomo model.

    **NOTE**: The formatting of the input data will be inferred using the same organizational scheme as ModelToDF. Please reference the ModelToDF documentation for details.

    Parameters
    ----------
    model: pyo.ConcreteModel
        The pyomo model you'd like to load data into
    data: dict
        A dict mapping the name of each variable in the model the a panadas dataframe containing the values to load for that variable
    """
    for readableVarName in data:
        if '.' in readableVarName:
            path,component = readableVarName.rsplit('.',1)
            varName = MergableModel.subModelKeyword + path.replace('.',MergableModel.componentKeyword+MergableModel.subModelKeyword) + MergableModel.componentKeyword + component
        else:
            varName = readableVarName
        var = getattr(model,varName)
        LoadVarSolutionFromDF(var,data[readableVarName])
