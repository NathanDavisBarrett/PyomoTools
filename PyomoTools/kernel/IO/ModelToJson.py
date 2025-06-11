import json

import pyomo.kernel as pmo
from .ModelToDict import ModelToDict

def ModelToJson(model:pmo.block,fileName:str,indent:int=4):
    dct = ModelToDict(model,reprKeys=True)

    with open(fileName, 'w') as outFile:
        json.dump(dct, outFile, indent=indent)
    