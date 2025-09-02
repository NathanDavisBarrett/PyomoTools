import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from copy import deepcopy
from enum import Enum


class LeastInfeasibleDefinition(Enum):
    """
    Options
    -------
    L1_Norm:
        Minimize the L1-norm of all constraint violations
    Num_Violated_Constrs:
        Minimize the total number of violated constraints (ignoring the degree to which they're violated.) (required "BigM" keyword argument indicating the maximum violation you'd like to consider.) (Note that this is a MUCH more expensive objective.)
    Sequential:
        A sequential application of Num_Violated_Constrs then L1_Norm.
    L2_Norm:
        Minimize the L2-norm of all constraint violations. This finds a center point between violated constraints but is inherently nonlinear
    """

    L1_Norm = 1
    Num_Violated_Constrs = 2
    Sequential = 3
    L2_Norm = 4


def FindLeastInfeasibleSolution(
    originalModel: pyo.ConcreteModel,
    solver,
    leastInfeasibleDefinition: LeastInfeasibleDefinition = LeastInfeasibleDefinition.L1_Norm,
    solver_args: tuple = (),
    solver_kwargs: dict = {},
    **kwargs,
):
    """
    Often you'll run into models that are infeasible even they they shouldn't be. Typically there are constraints that are modeled incorrectly.

    But almost all the time, solvers do not give you any information about what is making the problem infeasible. They just say "model is infeasible".

    This function will relax all constraints in the model using artificial slack variables (producing an "augmented" copy of the original model)

    Then, with the solver provided, this function will solve the augmented model to find the least infeasible solution.

    This solution will then be loaded back into the model object provided for further analysis (likely by the InfeasibilityReport class).

    #NOTE: This is not guaranteed to work unless all variable bounds are defined using only the "bounds" keyword in the variable definition and NOT using implicit bounds via the "domain" keyword (e.g. NOT pyo.NonNegativeReals). Pyomo makes it quite difficult to change these domains.

    Parameters:
    -----------
    originalModel: pyo.ConcreteModel
        The model you'd like to analyze
    solver: Pyomo SolverFactoryObject
        The solver you'd like to use to solve the augmented problem.
    leastInfeasibleDefinition: LeastInfeasibleDefinition (optional, Default = L1_Norm)
        The definition you'd like to use as "least" infeasible.
    solver_args: tuple (optional, Default = ())
        Any other arguments to pass to the solver's solve function.
    solver_kwargs: dict (optional, Default = {})
        Any other key-word arguments to pass to the solver's solve function.
    **kwargs: dict
        Other keyword arguments as needed by the leastInfeasibleDefinition
    """

    augmentedModel = deepcopy(originalModel)
    slackVars = []
    # Step 1, Change all variable bounds to explicit constraints (Thus they will have their slack variables added in step 2.)
    for var in augmentedModel.component_objects(pyo.Var, active=True):
        varName = str(var)
        isIndexed = "Indexed" in str(type(var))

        if isIndexed:

            def lowerBound(_, *idx):
                lb = var[idx].bounds[0]
                if lb is not None:
                    var[idx].setlb(None)
                    return lb <= var[idx]
                else:
                    return pyo.Constraint.Feasible

            def upperBound(_, *idx):
                ub = var[idx].bounds[1]
                if ub is not None:
                    var[idx].setub(None)
                    return ub >= var[idx]
                else:
                    return pyo.Constraint.Feasible

            setattr(
                augmentedModel,
                f"{var}_LOWER_BOUND",
                pyo.Constraint(var.index_set(), rule=lowerBound),
            )
            setattr(
                augmentedModel,
                f"{var}_UPPER_BOUND",
                pyo.Constraint(var.index_set(), rule=upperBound),
            )

        else:
            bounds = var.bounds
            if bounds[0] is not None:
                constrName = f"{var}_LOWER_BOUND"
                setattr(
                    augmentedModel, constrName, pyo.Constraint(expr=bounds[0] <= var)
                )
                var.setlb(None)
            if bounds[1] is not None:
                constrName = f"{var}_UPPER_BOUND"
                setattr(
                    augmentedModel, constrName, pyo.Constraint(expr=bounds[1] >= var)
                )
                var.setub(None)

    # Step 2, copy over all constraints, relaxing each one using additional slack variables.
    constrNames = [
        str(constr)
        for constr in augmentedModel.component_objects(pyo.Constraint, active=True)
    ]
    for constrName in constrNames:
        constr = getattr(augmentedModel, constrName)
        isIndexed = "Indexed" in str(type(constr))

        slackVarName = f"{constrName}_SLACK"

        if isIndexed:
            indexSet = constr.index_set()
            setattr(
                augmentedModel,
                slackVarName,
                pyo.Var(indexSet, domain=pyo.NonNegativeReals),
            )
            slackVar = getattr(augmentedModel, slackVarName)
            slackVars.extend(slackVar[idx] for idx in indexSet)
        else:
            setattr(augmentedModel, slackVarName, pyo.Var(domain=pyo.NonNegativeReals))
            slackVar = getattr(augmentedModel, slackVarName)
            slackVars.append(slackVar)

        if isIndexed:
            indexSet = constr.index_set()

            def lowerConstr(_, *idx):
                try:
                    constri = constr[idx]
                except Exception:
                    # Sometimes, if an iterate is nullified using pyo.Constraint.Feasible, it won't show up here.
                    return pyo.Constraint.Feasible

                if constri.lower is not None:
                    return constri.lower - slackVar[idx] <= constri.body
                else:
                    return pyo.Constraint.Feasible

            def upperConstr(_, *idx):
                try:
                    constri = constr[idx]
                except Exception:
                    # Sometimes, if an iterate is nullified using pyo.Constraint.Feasible, it won't show up here.
                    return pyo.Constraint.Feasible

                if constri.upper is not None:
                    return constri.body <= constri.upper + slackVar[idx]
                else:
                    return pyo.Constraint.Feasible

            lowerConstrName = f"{constrName}_LOWER_CONSTR"
            upperConstrName = f"{constrName}_UPPER_CONSTR"

            setattr(
                augmentedModel,
                lowerConstrName,
                pyo.Constraint(indexSet, rule=lowerConstr),
            )
            setattr(
                augmentedModel,
                upperConstrName,
                pyo.Constraint(indexSet, rule=upperConstr),
            )

        else:
            body = constr.body
            if constr.lower is not None:
                lowerConstrName = f"{constrName}_LOWER_CONSTR"
                setattr(
                    augmentedModel,
                    lowerConstrName,
                    pyo.Constraint(expr=constr.lower - slackVar <= body),
                )
            if constr.upper is not None:
                upperConstrName = f"{constrName}_UPPER_CONSTR"
                setattr(
                    augmentedModel,
                    upperConstrName,
                    pyo.Constraint(expr=body <= constr.upper + slackVar),
                )

        constr.deactivate()

    # Step 3: Deactivate all objectives
    for obj in augmentedModel.component_objects(pyo.Objective, active=True):
        obj.deactivate()

    # Step 4: Define the augmented objective.
    if leastInfeasibleDefinition == LeastInfeasibleDefinition.L1_Norm:
        augmentedModel.LEAST_INFEASIBLE_L1_OBJ = pyo.Objective(
            expr=sum(slackVars), sense=pyo.minimize
        )
    elif leastInfeasibleDefinition == LeastInfeasibleDefinition.L2_Norm:
        augmentedModel.LEAST_INFEASIBLE_L2_OBJ = pyo.Objective(
            expr=sum(s**2 for s in slackVars), sense=pyo.minimize
        )
    elif leastInfeasibleDefinition in [
        LeastInfeasibleDefinition.Num_Violated_Constrs,
        LeastInfeasibleDefinition.Sequential,
    ]:
        assert (
            "BigM" in kwargs
        ), 'The Num_Violated_Constrs requires a "BigM" parameter to be passed in.'
        BigM = kwargs["BigM"]
        indices = list(range(len(slackVars)))
        augmentedModel.slackActive = pyo.Var(indices, domain=pyo.Binary)

        augmentedModel.slackActive_Definition = pyo.Constraint(
            indices,
            rule=lambda _, i: slackVars[i] <= BigM * augmentedModel.slackActive[i],
        )

        augmentedModel.LEAST_INFEASIBLE_NUM_VIOLATED_OBJ = pyo.Objective(
            expr=sum(augmentedModel.slackActive[i] for i in indices), sense=pyo.minimize
        )
    else:
        raise Exception(
            f"{leastInfeasibleDefinition} is not a recognized definition. Please refer to options in the LeastInfeasibleDefinition enum."
        )

    # Step 5: Solve the augmented model.
    result = solver.solve(augmentedModel, *solver_args, **solver_kwargs)
    if result.solver.termination_condition != TerminationCondition.optimal:
        raise Exception(
            'Something has gone wrong. The problem is likely due to variable bounds being defined the "domain" keyword in the variable definition. Please try using "bounds" keyword there.'
        )

    if leastInfeasibleDefinition == LeastInfeasibleDefinition.Sequential:
        # Fix all slack vars that are not active.
        for i in indices:
            val = pyo.value(augmentedModel.slackActive[i])
            if val <= 0.5:
                slackVars[i].fix(0)
                augmentedModel.slackActive[i].fix(0)
            else:
                augmentedModel.slackActive[i].fix(1)

        augmentedModel.slackActive_Definition.deactivate()
        augmentedModel.LEAST_INFEASIBLE_NUM_VIOLATED_OBJ.deactivate()

        augmentedModel.LEAST_INFEASIBLE_L1_OBJ = pyo.Objective(
            expr=sum(slackVars), sense=pyo.minimize
        )

        result = solver.solve(augmentedModel, *solver_args, **solver_kwargs)
        if result.solver.termination_condition != TerminationCondition.optimal:
            raise Exception(
                'Something has gone wrong. The problem is likely due to variable bounds being defined the "domain" keyword in the variable definition. Please try using "bounds" keyword there.'
            )

    # Step 6: Copy the solution from the augmented model back to the original model.
    for var in originalModel.component_objects(pyo.Var, active=True):
        varName = str(var)
        isIndexed = "Indexed" in str(type(var))
        augmentedVar = getattr(augmentedModel, varName)
        if isIndexed:
            for idx in var.index_set():
                var[idx].value = augmentedVar[idx].value
        else:
            var.value = augmentedVar.value
