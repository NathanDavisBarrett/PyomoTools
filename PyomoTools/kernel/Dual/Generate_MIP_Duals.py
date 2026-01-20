"""
A module for computing the duals of MIP models by fixing integer variables and solving the remaining LP.
"""

import pyomo.kernel as pmo
from .IncorporateDuals import IncorporateDuals
from ...base.Solvers import DefaultSolver
from ..ParallelComponentIterator import ParallelComponentIterator
from typing import Optional


def _fix_integer_vars(model: pmo.block) -> None:
    for c in model.children():
        if isinstance(c, pmo.variable):
            if c.is_integer() or c.is_binary():
                c.fix(c.value)
        elif isinstance(c, (pmo.variable_list, pmo.variable_tuple)):
            for var in c:
                if var.is_integer() or var.is_binary():
                    var.fix(var.value)
        elif isinstance(c, pmo.variable_dict):
            for key in c:
                var = c[key]
                if var.is_integer() or var.is_binary():
                    var.fix(var.value)
        elif isinstance(c, pmo.block):
            _fix_integer_vars(c)
        elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
            for b in c:
                _fix_integer_vars(b)
        elif isinstance(c, pmo.block_dict):
            for key in c:
                b = c[key]
                _fix_integer_vars(b)


def Generate_MIP_Duals(model: pmo.block, solver=None) -> None:
    if solver is None:
        solver = DefaultSolver("LP")

    # Clone the original model
    lp_model = model.clone()

    # Fix integer variables
    _fix_integer_vars(lp_model)

    # Ensure dual suffix exists
    if not hasattr(lp_model, "dual"):
        lp_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    # Solve the LP relaxation
    solver.solve(lp_model)

    # Copy over duals to the original model
    if not hasattr(model, "dual"):
        model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    # for key in lp_model.dual:
    #     model.dual[key] = lp_model.dual[key]
    for o_constr, n_constr in ParallelComponentIterator(
        [model, lp_model],
        collect_vars=False,
        collect_constrs=True,
        collect_objs=False,
    ):
        if n_constr in lp_model.dual:
            model.dual[o_constr] = lp_model.dual[n_constr]

    # Disperse duals to child blocks
    IncorporateDuals(model)
