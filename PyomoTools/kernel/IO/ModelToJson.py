import json

import pyomo.kernel as pmo
from .ModelToDict import ModelToDict


def ModelToJson(model: pmo.block, fileName: str, indent: int = 4):
    dct = ModelToDict(model, reprKeys=True)

    with open(fileName, "w") as outFile:
        try:
            json.dump(dct, outFile, indent=indent)
        except TypeError as e:
            raise TypeError(
                f"Failed to serialize model to JSON: {e}. Ensure all objects in the model are JSON serializable.\nDict: {dct}"
            ) from e
