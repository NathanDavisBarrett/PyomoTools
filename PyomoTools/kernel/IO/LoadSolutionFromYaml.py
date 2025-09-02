import yaml

import pyomo.kernel as pmo
from .LoadSolutionFromDict import LoadSolutionFromDict


def LoadSolutionFromYaml(model: pmo.block, fileName: str):
    with open(fileName, "r") as inFile:
        reprdct = yaml.safe_load(inFile)
    LoadSolutionFromDict(model, reprdct, unRepr=True)
