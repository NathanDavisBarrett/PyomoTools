import pyomo.kernel as pmo
from warnings import warn
from enum import Enum

from .FormatKey import FormatKey


class AnomalyOutcome(Enum):
    Error = 0
    Warning = 1
    Ignore = 2


def HandleAnomaly(message, outcome: AnomalyOutcome):
    if outcome == AnomalyOutcome.Error:
        raise Exception(message)
    elif outcome == AnomalyOutcome.Warning:
        warn(message)
    else:
        pass


def LoadSolutionFromDict(
    model: pmo.block,
    dct: dict,
    anomalyOutcome: AnomalyOutcome = AnomalyOutcome.Error,
    unRepr: bool = True,
):
    for c in model.children():
        if isinstance(
            c,
            (
                pmo.constraint,
                pmo.constraint_list,
                pmo.constraint_tuple,
                pmo.constraint_dict,
                pmo.objective,
                pmo.objective_list,
                pmo.objective_tuple,
                pmo.objective_dict,
                pmo.expression,
                pmo.expression_list,
                pmo.expression_tuple,
                pmo.expression_dict,
            ),
        ):
            # Constraints and objectives are not loaded from the solution dict
            continue

        cName = c.local_name
        cNameDict = FormatKey(repr(cName) if unRepr else cName)

        if cNameDict in dct:
            e = dct[cNameDict]
        else:
            HandleAnomaly(
                f'Model component "{cName}" could not be found in the provided data. Unless an error is thrown here, None will be used for all variable values resulting from this anomaly',
                anomalyOutcome,
            )
            e = None

        if isinstance(c, (pmo.variable_list, pmo.variable_tuple)):
            if e is not None:
                if not isinstance(e, list):
                    HandleAnomaly(
                        f'List-type model component"{cName}" did not correspond with a list in the provided data. Unless an error is thrown here, None will be used.',
                        anomalyOutcome,
                    )
                    for i in range(len(c)):
                        c[i].value = None
                elif len(c) != len(e):
                    HandleAnomaly(
                        f'The length of variable list "{cName}" provided by the data ({len(e)}) does not match that found in the model ({len(c)}). Unless an error is thrown here, None will be used.',
                        anomalyOutcome,
                    )
                    for i in range(len(c)):
                        c[i].value = None
                else:
                    for i in range(len(c)):
                        c[i].value = e[i]
            else:
                for i in range(len(c)):
                    c[i].value = None
        elif isinstance(c, pmo.variable_dict):
            if e is not None:
                if not isinstance(e, dict):
                    HandleAnomaly(
                        f'Dict-type model component"{cName}" did not correspond with a dict in the provided data. Unless an error is thrown here, None will be used.',
                        anomalyOutcome,
                    )
                    for k in c:
                        c[k].value = None
                else:
                    for k in c:
                        kDict = FormatKey(
                            repr(k) if unRepr else k
                        )  # repr(k) if unRepr else k
                        if kDict not in e:
                            HandleAnomaly(
                                f'Key "{k}" from model component "{cName}" was not found in the data. Unless an error is thrown here, None will be used.',
                                anomalyOutcome,
                            )
                            c[k].value = None
                        else:
                            c[k].value = e[kDict]
            else:
                for k in c:
                    c[k].value = None
        elif isinstance(c, pmo.variable):
            c.value = e

        elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
            if e is not None:
                if not isinstance(e, list):
                    HandleAnomaly(
                        f'List-type model component"{cName}" did not correspond with a list in the provided data. Unless an error is thrown here, None will be used.',
                        anomalyOutcome,
                    )
                    for i in range(len(c)):
                        LoadSolutionFromDict(
                            c[i], {}, AnomalyOutcome.Ignore, unRepr=unRepr
                        )
                elif len(c) != len(e):
                    HandleAnomaly(
                        f'The length of variable list "{cName}" provided by the data ({len(e)}) does not match that found in the model ({len(c)}). Unless an error is thrown here, None will be used.',
                        anomalyOutcome,
                    )
                    for i in range(len(c)):
                        LoadSolutionFromDict(
                            c[i], {}, AnomalyOutcome.Ignore, unRepr=unRepr
                        )
                else:
                    for i in range(len(c)):
                        LoadSolutionFromDict(c[i], e[i], anomalyOutcome, unRepr=unRepr)
            else:
                for i in range(len(c)):
                    LoadSolutionFromDict(c[i], {}, AnomalyOutcome.Ignore, unRepr=unRepr)
        elif isinstance(c, pmo.block_dict):
            if e is not None:
                if not isinstance(e, dict):
                    HandleAnomaly(
                        f'Dict-type model component"{cName}" did not correspond with a dict in the provided data. Unless an error is thrown here, None will be used.',
                        anomalyOutcome,
                    )
                    for k in c:
                        LoadSolutionFromDict(
                            c[k], {}, AnomalyOutcome.Ignore, unRepr=unRepr
                        )
                else:
                    for k in c:
                        kDict = FormatKey(
                            repr(k) if unRepr else k
                        )  # repr(k) if unRepr else k
                        if kDict not in e:
                            HandleAnomaly(
                                f'Key "{k}" from model component "{cName}" was not found in the data. Unless an error is thrown here, None will be used.',
                                anomalyOutcome,
                            )
                            LoadSolutionFromDict(
                                c[k], {}, AnomalyOutcome.Ignore, unRepr=unRepr
                            )
                        else:
                            LoadSolutionFromDict(
                                c[k], e[kDict], anomalyOutcome, unRepr=unRepr
                            )
            else:
                for k in c:
                    LoadSolutionFromDict(c[k], {}, AnomalyOutcome.Ignore, unRepr=unRepr)
        elif isinstance(c, pmo.block):
            if e is not None:
                LoadSolutionFromDict(c, e, anomalyOutcome, unRepr=unRepr)
            else:
                LoadSolutionFromDict(c, {}, AnomalyOutcome.Ignore, unRepr=unRepr)
