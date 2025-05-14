import yaml

import pyomo.kernel as pmo
from .ModelToDict import ModelToDict

def ModelToYaml(model:pmo.block,fileName:str):
    dct = ModelToDict(model)

    with open(fileName, 'w') as outFile:
        yaml.safe_dump(dct, outFile, default_flow_style=False)
    