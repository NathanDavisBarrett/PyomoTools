import pyomo.environ as pyo


def Load2DIndexedSet(model, setName, setDict):
    """
    A function to define a 2-leveled indexed set (dict mapping keys to another dict mapping keys to pyomo Set objects) within a pyomo model.

    Once defined you will be able to access each subset using the following syntax:
        model.setName[index1][index2]

    Parameters
    ----------
    model: pyo.ConcreteModel
        The model to which you'd like to add the indexed set
    setName: str
        The name of the set you'd like to add. Node that this needs to friendly to python syntax (e.g. no spaces, periods, dashes, etc.)
    setDict: dict (non-iterable -> iterable)
        A dict mapping each key to another dict mapping each sub-key the contents of each corresponding subset.
    """
    setattr(model, setName, {})
    for k1 in setDict:
        getattr(model, setName)[k1] = {}
        for k2 in setDict[k1]:
            attrName = "{}_{}_{}".format(setName, k1, k2)
            setattr(model, attrName, pyo.Set(initialize=setDict[k1][k2]))
            getattr(model, setName)[k1][k2] = getattr(model, attrName)
