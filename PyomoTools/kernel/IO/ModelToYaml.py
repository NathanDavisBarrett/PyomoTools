import yaml

def represent_tuple(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data)

yaml.add_representer(tuple, represent_tuple)

import pyomo.kernel as pmo
from .ModelToDict import ModelToDict

def ModelToYaml(model:pmo.block,fileName:str):
    dct = ModelToDict(model,reprKeys=True)

    with open(fileName, 'w') as outFile:
        yaml.safe_dump(dct, outFile, default_flow_style=False)
    