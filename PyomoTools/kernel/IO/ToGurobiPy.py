try:
    import gurobipy
except ImportError:
    print("gurobipy not found. Installing...")
    try:
        import pip

        pip.main(["install", "gurobipy"])
        import gurobipy

        print("gurobipy installed successfully.")
    except Exception as e:
        print(f"Error installing gurobipy: {e}")
        print("Please install gurobipy manually using 'pip install gurobipy'")

import pyomo.kernel as pmo
from pyomo.core.expr.current import identify_variables
import numpy as np


def _GetDomain(var: pmo.variable):
    if var._domain_type.is_continuous():
        return gurobipy.GRB.CONTINUOUS
    elif var._domain_type.is_integer():
        return gurobipy.GRB.INTEGER
    elif var._domain_type.is_binary():
        return gurobipy.GRB.BINARY
    else:
        raise ValueError(f"Unable to determine domain type: {var._domain_type}")


nameSpaceDelimiter = "_BLK_"


def _CopyVariables(grbModel, model, nameSpacer=""):
    # TODO: Set initial values for variables if they exist in the model
    variableMap = {}
    for c in model.children():
        if isinstance(c, pmo.variable):
            vr = grbModel.addVar(
                name=nameSpacer + c.local_name,
                lb=c.lb if c.lb is not None else -gurobipy.GRB.INFINITY,
                ub=c.ub if c.ub is not None else gurobipy.GRB.INFINITY,
                vtype=_GetDomain(c),
            )
            if c.value is not None:
                vr.start = c.value
            variableMap[c.name] = vr
        elif isinstance(c, (pmo.variable_list, pmo.variable_tuple)):
            vr = grbModel.addVars(
                list(range(len(c))),
                name=nameSpacer + c.local_name,
                lb=[v.lb if v.lb is not None else -gurobipy.GRB.INFINITY for v in c],
                ub=[v.ub if v.ub is not None else gurobipy.GRB.INFINITY for v in c],
                vtype=[_GetDomain(v) for v in c],
            )
            for i, v in enumerate(c):
                if v.value is not None:
                    vr[i].start = v.value
                variableMap[v.name] = vr[i]
        elif isinstance(c, pmo.variable_dict):
            vr = grbModel.addVars(
                list(c.keys()),
                name=nameSpacer + c.local_name,
                lb={
                    k: (v.lb if v.lb is not None else -gurobipy.GRB.INFINITY)
                    for k, v in c.items()
                },
                ub={
                    k: (v.ub if v.ub is not None else gurobipy.GRB.INFINITY)
                    for k, v in c.items()
                },
                vtype={k: _GetDomain(v) for k, v in c.items()},
            )
            for k, v in c.items():
                if v.value is not None:
                    vr[k].start = v.value
                variableMap[v.name] = vr[k]

        elif isinstance(c, pmo.block):
            subMap = _CopyVariables(
                grbModel, c, nameSpacer + nameSpaceDelimiter + c.local_name
            )
            variableMap.update(subMap)

        elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
            for i, b in enumerate(c):
                subMap = _CopyVariables(
                    grbModel,
                    b,
                    nameSpacer + nameSpaceDelimiter + f"[{i}]" + b.local_name,
                )
                variableMap.update(subMap)
        elif isinstance(c, pmo.block_dict):
            for k, b in c.items():
                subMap = _CopyVariables(
                    grbModel,
                    b,
                    nameSpacer + nameSpaceDelimiter + f"[{k}]" + b.local_name,
                )
                variableMap.update(subMap)
    return variableMap


def _ConvertExpression(expr: pmo.expression, varMap: dict):
    vrs = list(identify_variables(expr))

    exprStr = str(expr)
    for v in vrs:
        vrStr = v.name
        exprStr = exprStr.replace(vrStr, f"varMap['{vrStr}']")

    return eval(exprStr)


def _ConvertRelational(c: pmo.constraint, varMap: dict):
    body = c.body
    ub = c.upper
    lb = c.lower

    if ub is not None and lb is not None:
        assert np.allclose(
            ub, lb
        ), "Upper and lower bounds must be equal for equality constraints."
        expr = _ConvertExpression(body, varMap) == ub
    elif ub is not None:
        expr = _ConvertExpression(body, varMap) <= ub
    elif lb is not None:
        expr = _ConvertExpression(body, varMap) >= lb
    else:
        raise ValueError(
            "At least one of upper or lower bounds must be specified for a constraint."
        )
    return expr


def _CopyConstraints(grbModel: gurobipy.Model, model: pmo.block, varMap, nameSpacer=""):
    for c in model.children():
        if isinstance(c, pmo.constraint):
            if not c.active:
                continue
            grbModel.addConstr(
                _ConvertRelational(c, varMap), name=nameSpacer + c.local_name
            )
        elif isinstance(c, (pmo.constraint_list, pmo.constraint_tuple)):
            grbModel.addConstrs(
                (_ConvertRelational(cc, varMap) for cc in c if cc.active),
                name=nameSpacer + c.local_name,
            )
        elif isinstance(c, pmo.constraint_dict):
            grbModel.addConstrs(
                (_ConvertRelational(cc, varMap) for cc in c.values() if cc.active),
                name=nameSpacer + c.local_name,
            )

        elif isinstance(c, pmo.block):
            _CopyConstraints(
                grbModel, c, varMap, nameSpacer + nameSpaceDelimiter + c.local_name
            )
        elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
            for i, b in enumerate(c):
                _CopyConstraints(
                    grbModel,
                    b,
                    varMap,
                    nameSpacer + nameSpaceDelimiter + f"[{i}]" + b.local_name,
                )
        elif isinstance(c, pmo.block_dict):
            for k, b in c.items():
                _CopyConstraints(
                    grbModel,
                    b,
                    varMap,
                    nameSpacer + nameSpaceDelimiter + f"[{k}]" + b.local_name,
                )


def _CopyObjectives(
    grbModel: gurobipy.Model, model: pmo.block, varMap: dict, nameSpacer=""
):
    for c in model.children():
        if isinstance(c, pmo.objective):
            if not c.active:
                continue
            grbModel.setObjective(
                _ConvertExpression(c.expr, varMap),
                sense=(
                    gurobipy.GRB.MINIMIZE
                    if c.sense == pmo.minimize
                    else gurobipy.GRB.MAXIMIZE
                ),
            )
        elif isinstance(c, (pmo.objective_list, pmo.objective_tuple)):
            raise NotImplementedError(
                "Objective lists and tuples are not supported in GurobiPy conversion."
            )
        elif isinstance(c, pmo.objective_dict):
            raise NotImplementedError(
                "Objective dictionaries are not supported in GurobiPy conversion."
            )

        elif isinstance(c, pmo.block):
            _CopyObjectives(
                grbModel, c, varMap, nameSpacer + nameSpaceDelimiter + c.local_name
            )
        elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
            for i, b in enumerate(c):
                _CopyObjectives(
                    grbModel,
                    b,
                    varMap,
                    nameSpacer + nameSpaceDelimiter + f"[{i}]" + b.local_name,
                )
        elif isinstance(c, pmo.block_dict):
            for k, b in c.items():
                _CopyObjectives(
                    grbModel,
                    b,
                    varMap,
                    nameSpacer + nameSpaceDelimiter + f"[{k}]" + b.local_name,
                )


def ToGurobiPy(model: pmo.block) -> gurobipy.Model:
    name = model.local_name
    if name is None or name == "":
        name = "PyomoModel"
    grbModel = gurobipy.Model(name)

    varMap = _CopyVariables(grbModel, model)
    _CopyConstraints(grbModel, model, varMap)
    _CopyObjectives(grbModel, model, varMap)

    return grbModel
