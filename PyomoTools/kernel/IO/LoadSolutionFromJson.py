import json

import pyomo.kernel as pmo
from .LoadSolutionFromDict import LoadSolutionFromDict


def LoadSolutionFromJson(model: pmo.block, fileName: str):
    with open(fileName, "r") as inFile:
        dct = json.load(inFile)
    LoadSolutionFromDict(model, dct, unRepr=True)
