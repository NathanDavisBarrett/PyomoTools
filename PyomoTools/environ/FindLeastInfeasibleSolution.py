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


def _is_indexed(component):
    """Check if a pyomo component is indexed."""
    return hasattr(component, "index_set") and component.is_indexed()


def _get_component_name(component):
    """Get the full name path of a component."""
    return component.name


def MapSpecificConstraint(originalModel, augmentedModel, component_name):
    """
    Map a component from the original model to the corresponding component in the augmented model.

    Parameters:
    -----------
    originalModel: pyo.ConcreteModel
        The original model
    augmentedModel: pyo.ConcreteModel
        The augmented (deepcopy'd) model
    component_name: str
        The full name path of the component (e.g., "x", "sub.c1", "c[A]")

    Returns:
    --------
    The corresponding component in the augmented model
    """
    # Split the name by dots to handle nested blocks
    name_parts = component_name.split(".")

    original_current = originalModel
    augmented_current = augmentedModel

    for part in name_parts:
        # Check if this part has an index (e.g., "c[A]" or "sub[1]")
        if "[" in part:
            base_name = part.split("[")[0]
            index_str = part.split("[")[1].rstrip("]")

            original_component = getattr(original_current, base_name)
            augmented_component = getattr(augmented_current, base_name)

            # Find the actual index by comparing string representations
            actual_index = None
            for key in original_component:
                if str(key) == index_str:
                    actual_index = key
                    break

            if actual_index is None:
                # Try interpreting as integer for simple indexed sets
                try:
                    actual_index = int(index_str)
                except ValueError:
                    raise ValueError(
                        f"Could not find index '{index_str}' in {base_name}"
                    )

            original_current = original_component[actual_index]
            augmented_current = augmented_component[actual_index]
        else:
            original_current = getattr(original_current, part)
            augmented_current = getattr(augmented_current, part)

    return augmented_current


def MapSpecificConstraints(
    originalModel, augmentedModel, relax_only_these_constraints
) -> list:
    """
    Map a list of components from the original model to the augmented model.

    Parameters:
    -----------
    originalModel: pyo.ConcreteModel
        The original model
    augmentedModel: pyo.ConcreteModel
        The augmented (deepcopy'd) model
    relax_only_these_constraints: list or None
        List of components to map

    Returns:
    --------
    List of mapped components in the augmented model, or None if input is None
    """
    if relax_only_these_constraints is None:
        return None

    mapped_constraints = []
    for target in relax_only_these_constraints:
        mapped_target = MapSpecificConstraint(
            originalModel, augmentedModel, _get_component_name(target)
        )
        mapped_constraints.append(mapped_target)
    return mapped_constraints


def _configure_constraint_relaxation(
    augmentedModel, constr, slackVars, constrName, isIndexed
):
    """
    Configure relaxation for a single constraint or indexed constraint.
    Creates slack variables and relaxed constraints.
    """
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
                return pyo.Constraint.Feasible

            if constri.lower is not None:
                return constri.lower - slackVar[idx] <= constri.body
            else:
                return pyo.Constraint.Feasible

        def upperConstr(_, *idx):
            try:
                constri = constr[idx]
            except Exception:
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


def _configure_variable_bound_relaxation(
    augmentedModel, var, slackVars, varName, isIndexed
):
    """
    Configure relaxation for variable bounds.
    Creates slack variables and relaxed bound constraints.
    """
    if isIndexed:
        indexSet = var.index_set()

        # Create slack variables for lower bounds
        lbSlackVarName = f"{varName}_LB_SLACK"
        setattr(
            augmentedModel,
            lbSlackVarName,
            pyo.Var(indexSet, domain=pyo.NonNegativeReals),
        )
        lbSlackVar = getattr(augmentedModel, lbSlackVarName)

        # Create slack variables for upper bounds
        ubSlackVarName = f"{varName}_UB_SLACK"
        setattr(
            augmentedModel,
            ubSlackVarName,
            pyo.Var(indexSet, domain=pyo.NonNegativeReals),
        )
        ubSlackVar = getattr(augmentedModel, ubSlackVarName)

        def lowerBound(_, *idx):
            lb = var[idx].bounds[0]
            if lb is not None:
                var[idx].setlb(None)
                slackVars.append(lbSlackVar[idx])
                return lb - lbSlackVar[idx] <= var[idx]
            else:
                return pyo.Constraint.Feasible

        def upperBound(_, *idx):
            ub = var[idx].bounds[1]
            if ub is not None:
                var[idx].setub(None)
                slackVars.append(ubSlackVar[idx])
                return var[idx] <= ub + ubSlackVar[idx]
            else:
                return pyo.Constraint.Feasible

        setattr(
            augmentedModel,
            f"{varName}_LOWER_BOUND_RELAXED",
            pyo.Constraint(indexSet, rule=lowerBound),
        )
        setattr(
            augmentedModel,
            f"{varName}_UPPER_BOUND_RELAXED",
            pyo.Constraint(indexSet, rule=upperBound),
        )
    else:
        bounds = var.bounds
        if bounds[0] is not None:
            slackVarName = f"{varName}_LB_SLACK"
            setattr(augmentedModel, slackVarName, pyo.Var(domain=pyo.NonNegativeReals))
            slackVar = getattr(augmentedModel, slackVarName)
            slackVars.append(slackVar)
            constrName = f"{varName}_LOWER_BOUND_RELAXED"
            setattr(
                augmentedModel,
                constrName,
                pyo.Constraint(expr=bounds[0] - slackVar <= var),
            )
            var.setlb(None)
        if bounds[1] is not None:
            slackVarName = f"{varName}_UB_SLACK"
            setattr(augmentedModel, slackVarName, pyo.Var(domain=pyo.NonNegativeReals))
            slackVar = getattr(augmentedModel, slackVarName)
            slackVars.append(slackVar)
            constrName = f"{varName}_UPPER_BOUND_RELAXED"
            setattr(
                augmentedModel,
                constrName,
                pyo.Constraint(expr=var <= bounds[1] + slackVar),
            )
            var.setub(None)


def _augment_all_constraints(augmentedModel):
    """
    Augment all constraints and variable bounds in the model.
    Returns the list of slack variables.
    """
    slackVars = []

    # Step 1, Change all variable bounds to explicit constraints
    for var in augmentedModel.component_objects(pyo.Var, active=True):
        varName = str(var)
        isIndexed = _is_indexed(var)

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
        isIndexed = _is_indexed(constr)
        _configure_constraint_relaxation(
            augmentedModel, constr, slackVars, constrName, isIndexed
        )

    # Step 3: Deactivate all objectives
    for obj in augmentedModel.component_objects(pyo.Objective, active=True):
        obj.deactivate()

    return slackVars


def _augment_specific_constraints(augmentedModel, relax_only_these_constraints):
    """
    Augment only specific constraints and/or variable bounds.
    Returns the list of slack variables.
    """
    slackVars = []
    blocks_to_fully_augment = []

    for target in relax_only_these_constraints:
        # Check if this is a Block (ConcreteModel or sub-block)
        if isinstance(target, pyo.Block):
            blocks_to_fully_augment.append(target)
        # Check if this is a Var or indexed Var
        elif isinstance(target, pyo.Var):
            varName = str(target)
            isIndexed = _is_indexed(target)
            _configure_variable_bound_relaxation(
                augmentedModel, target, slackVars, varName, isIndexed
            )
        # Check if this is a Constraint or indexed Constraint
        elif isinstance(target, pyo.Constraint):
            constrName = str(target)
            isIndexed = _is_indexed(target)
            _configure_constraint_relaxation(
                augmentedModel, target, slackVars, constrName, isIndexed
            )

    # For blocks, we need to augment all constraints within them
    for block in blocks_to_fully_augment:
        # Augment all variable bounds in this block
        for var in block.component_objects(pyo.Var, active=True, descend_into=True):
            varName = str(var)
            isIndexed = _is_indexed(var)
            _configure_variable_bound_relaxation(
                augmentedModel, var, slackVars, varName, isIndexed
            )

        # Augment all constraints in this block
        for constr in block.component_objects(
            pyo.Constraint, active=True, descend_into=True
        ):
            constrName = str(constr)
            isIndexed = _is_indexed(constr)
            _configure_constraint_relaxation(
                augmentedModel, constr, slackVars, constrName, isIndexed
            )

    # Deactivate all objectives (we still need to do this)
    for obj in augmentedModel.component_objects(pyo.Objective, active=True):
        obj.deactivate()

    return slackVars


def AugmentModel(augmentedModel, relax_only_these_constraints=None):
    """
    Augment the model by adding slack variables to relax constraints.

    Parameters:
    -----------
    augmentedModel: pyo.ConcreteModel
        The model to augment
    relax_only_these_constraints: list or None
        If provided, only these constraints/variables/blocks will be relaxed.
        Otherwise, all constraints will be relaxed.

    Returns:
    --------
    List of slack variables
    """
    if relax_only_these_constraints is None:
        return _augment_all_constraints(augmentedModel)
    else:
        return _augment_specific_constraints(
            augmentedModel, relax_only_these_constraints
        )


def FindLeastInfeasibleSolution(
    originalModel: pyo.ConcreteModel,
    solver,
    leastInfeasibleDefinition: LeastInfeasibleDefinition = LeastInfeasibleDefinition.L1_Norm,
    solver_args: tuple = (),
    solver_kwargs: dict = {},
    relax_only_these_constraints: list = None,
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
    relax_only_these_constraints: list (optional, Default = None)
        If provided, only these constraints (Constraint, Var (bounds will be relaxed), or Block objects) will be relaxed in the augmented model. Otherwise, all constraints will be relaxed.
    **kwargs: dict
        Other keyword arguments as needed by the leastInfeasibleDefinition
    """

    augmentedModel = deepcopy(originalModel)

    # Map the specific constraints from original to augmented model
    mapped_constraints = MapSpecificConstraints(
        originalModel, augmentedModel, relax_only_these_constraints
    )

    slackVars = AugmentModel(augmentedModel, mapped_constraints)
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
    if result.solver.termination_condition not in [
        TerminationCondition.optimal,
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
        TerminationCondition.minFunctionValue,
        TerminationCondition.minStepLength,
        TerminationCondition.globallyOptimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.maxEvaluations,
    ]:
        message = f'The solver terminated with condition "{result.solver.termination_condition}".'
        if relax_only_these_constraints is not None:
            message += " Note that only a subset of constraints were relaxed in this analysis. Even with these constraints relaxed, the solver could not find a feasible solution to the augmented problem. Please try relaxing more or all constraints."
        else:
            message += " This is caused by an unknown issue. Please report it as a bug."
        raise Exception(message)

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
            message = f'The solver terminated with condition "{result.solver.termination_condition}".'
            if relax_only_these_constraints is not None:
                message += " Note that only a subset of constraints were relaxed in this analysis. Even with these constraints relaxed, the solver could not find a feasible solution to the augmented problem. Please try relaxing more or all constraints."
            else:
                message += (
                    " This is caused by an unknown issue. Please report it as a bug."
                )
            raise Exception(message)

    # Step 6: Copy the solution from the augmented model back to the original model.
    for var in originalModel.component_objects(pyo.Var, active=True):
        varName = str(var)
        isIndexed = _is_indexed(var)
        augmentedVar = getattr(augmentedModel, varName)
        if isIndexed:
            for idx in var.index_set():
                var[idx].value = augmentedVar[idx].value
        else:
            var.value = augmentedVar.value
