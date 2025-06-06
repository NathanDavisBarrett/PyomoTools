import yaml

def represent_tuple(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data)

yaml.add_representer(tuple, represent_tuple)

import pyomo.kernel as pmo
from .LoadSolutionFromDict import LoadSolutionFromDict


def LoadSolutionFromYaml(model:pmo.block,fileName:str):
    with open(fileName, 'r') as inFile:
        dct = yaml.safe_load(inFile)
    LoadSolutionFromDict(model,dct)
    