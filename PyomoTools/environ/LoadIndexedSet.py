import pyomo.environ as pyo


def LoadIndexedSet(model: pyo.ConcreteModel, setName: str, setDict: dict):
    """
    A function to define an indexed set (dict mapping keys to pyomo Set objects) within a pyomo model.

    Once defined you will be able to access each subset using the following syntax:
        model.setName[index]

    Parameters
    ----------
    model: pyo.ConcreteModel
        The model to which you'd like to add the indexed set
    setName: str
        The name of the set you'd like to add. Node that this needs to friendly to python syntax (e.g. no spaces, periods, dashes, etc.)
    setDict: dict (index -> iterable)
        A dict mapping each key to the contents of each corresponding subset.
    """
    setattr(model, setName, {})
    for key in setDict:
        attrName = "{}_{}".format(setName, key)
        setattr(model, attrName, pyo.Set(initialize=setDict[key]))
        getattr(model, setName)[key] = getattr(model, attrName)
