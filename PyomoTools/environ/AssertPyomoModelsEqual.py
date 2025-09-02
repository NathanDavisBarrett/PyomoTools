import pyomo.environ as pyo
import numpy as np
import warnings


def SendErrorWarning(str, error, warning):
    if error:
        raise AssertionError(str)
    if warning:
        warnings.warn(str)


def AssertPyomoModelsEqual(
    model1: pyo.ConcreteModel,
    model2: pyo.ConcreteModel,
    rtol=1e-5,
    atol=1e-8,
    error=False,
    warning=False,
) -> bool:
    """
    A function to determine if two pyomo.environ models contain equal solutions.

    An equal solution is defined as follows.
        A) Both models have the same variables with the same names
        B) Each variable's value within model1 is within an "rtol" relative tolerance OR an "atol" absolute tolerance to that same variable's value in model 2.

    Parameters
    ----------
    model1: pyo.ConcreteModel
        The 1st pyomo model you'd like to compare
    model2: pyo.ConcreteModel
        The 2nd pyomo model you'd like to compare
    rtol: float (optional, Default: 1e-5)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    atol: float (optional, Default: 1e-8)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    error: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to throw an error if a mismatch is found.
    warning: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to raise a warning if a mismatch is found.

    Returns
    -------
    bool:
        A boolean indicating whether or not the two models are equal
    """
    variableNames1 = set([])
    for v1 in model1.component_objects():
        if v1.type() is pyo.Var:
            variableNames1.add(v1.getname())

    variableNames2 = set([])
    for v2 in model2.component_objects():
        if v2.type() is pyo.Var:
            variableNames2.add(v2.getname())

    if variableNames1 != variableNames2:
        if error or warning:
            variablesIn1ButNot2 = variableNames1 - variableNames2
            variablesIn2ButNot1 = variableNames2 - variableNames1
            SendErrorWarning(
                "The following variables are present in model1 but not model2:\n{}\n\nThe following variables are present in model2 but not model1:\n{}".format(
                    variablesIn1ButNot2, variablesIn2ButNot1
                ),
                error,
                warning,
            )
        return False

    for v in variableNames1:
        v1 = model1.find_component(v)
        v2 = model2.find_component(v)

        v1Indices = set([i for i in v1])
        v2Indices = set([i for i in v2])

        if v1Indices != v2Indices:
            if error or warning:
                SendErrorWarning(
                    f'The indices for variable "{v}" are not consistent between the two models.\nModel 1 Indices: {v1Indices}\nModel 2 Indices: {v2Indices}',
                    error,
                    warning,
                )
            return False

        for index in v1:
            val1 = v1[index].value
            val2 = v2[index].value

            if val1 is None or val2 is None:
                if val1 is not None or val2 is not None:
                    if error or warning:
                        SendErrorWarning(
                            "The values for {}[{}] do not match! model1: {}, model2: {}".format(
                                v, index, val1, val2
                            ),
                            error,
                            warning,
                        )
                    return False
                continue

            if val1 == 0:
                if val2 != 0:
                    if np.abs(val2) < atol:
                        continue
                    if error or warning:
                        SendErrorWarning(
                            "The values for {}[{}] do not match! model1: {}, model2: {}".format(
                                v, index, val1, val2
                            ),
                            error,
                            warning,
                        )
                    return False
            else:
                aerr = np.abs(val1 - val2)
                rerr = np.abs((val1 - val2) / val1)
                if not (rerr < rtol or aerr < atol):
                    if error or warning:
                        SendErrorWarning(
                            "The values for {}[{}] do not match! model1: {}, model2: {}".format(
                                v, index, val1, val2
                            ),
                            error,
                            warning,
                        )
                    return False

    return True
