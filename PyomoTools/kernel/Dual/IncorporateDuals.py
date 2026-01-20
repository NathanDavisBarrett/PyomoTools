"""
As it stands at the moment of writing, Pyomo kernel only supports dual suffixes on the parent-most block of a model.

This module provides functionality to incorporate dual values throughout child blocks of a model.
"""

import pyomo.kernel as pmo
import warnings
from typing import List, Union

_warning_issued = False


def warn(constr: pmo.constraint) -> None:
    global _warning_issued
    if not _warning_issued:
        warnings.warn(
            f"Dual value for constraint {constr} could not be found in the provided dual suffix. Please ensure the dual suffix is instantiated on the parent-most block of the model: parent_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)"
        )
        _warning_issued = True


def _IncorporateDuals(model: pmo.block, dual_dict: pmo.suffix) -> None:
    if not hasattr(model, "dual"):
        model.dual = pmo.suffix()

    for c in model.children():
        if isinstance(c, pmo.constraint):
            if c in dual_dict:
                model.dual[c] = dual_dict[c]
            else:
                warn(c)
        elif isinstance(c, (pmo.constraint_list, pmo.constraint_tuple)):
            for constr in c:
                if constr in dual_dict:
                    model.dual[constr] = dual_dict[constr]
                else:
                    warn(constr)
        elif isinstance(c, pmo.constraint_dict):
            for key in c:
                constr = c[key]
                if constr in dual_dict:
                    model.dual[constr] = dual_dict[constr]
                else:
                    warn(constr)
        elif isinstance(c, pmo.block):
            _IncorporateDuals(c, dual_dict)
        elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
            for b in c:
                _IncorporateDuals(b, dual_dict)
        elif isinstance(c, pmo.block_dict):
            for key in c:
                b = c[key]
                _IncorporateDuals(b, dual_dict)


def IncorporateDuals(model: pmo.block) -> None:
    if not hasattr(model, "dual"):
        raise ValueError(
            "The model does not have a dual suffix to incorporate. Please ensure the dual suffix is instantiated on the parent-most block of the model: parent_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT) and that the parent-most block has been solved and passed to this function."
        )
    _IncorporateDuals(model, model.dual)
