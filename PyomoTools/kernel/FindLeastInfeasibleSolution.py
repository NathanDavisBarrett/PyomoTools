import pyomo.kernel as pmo
from pyomo.opt import TerminationCondition
from enum import Enum
from collections import deque


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


def ConfigureConstraint(constr: pmo.constraint):
    # Step 1, create a slack variable for this constraint
    slackVar = pmo.variable(domain=pmo.NonNegativeReals)

    # Step 2, create a lower bound constraint
    if constr.lower is not None:
        lower = pmo.constraint(constr.lower - slackVar <= constr.body)
    else:
        lower = None

    # Step 3, create an upper bound constraint
    if constr.upper is not None:
        upper = pmo.constraint(constr.body <= constr.upper + slackVar)
    else:
        upper = None

    # Step 4, deactivate the original constraint
    constr.deactivate()

    return slackVar, lower, upper


def AugmentModel(model: pmo.block):
    # Step 1, Change all variable bounds to explicit constraints (Thus they will have their slack variables added in step 2.)
    # TODO: Recognize and change over domain types.
    # Step 2, Detect and deactivate an objectives.
    # Step 3, copy over all constraints, relaxing each one using additional slack variables.
    slackVars = deque([])  # Deque used for ultra-fast append operations.
    lowerBoundConstrs = deque([])
    upperBoundConstrs = deque([])

    def lowerBound(var):
        lb = var.bounds[0]
        if lb is not None:
            var.lb = None
            return lb <= var
        else:
            return None

    def upperBound(var):
        ub = var.bounds[1]
        if ub is not None:
            var.ub = None
            return ub >= var
        else:
            return None

    for c in model.children():
        if isinstance(c, (pmo.variable_list, pmo.variable_tuple)):
            lowerBoundConstrs.extend(
                [pmo.constraint(lowerBound(c[i])) for i in range(len(c))]
            )
            upperBoundConstrs.extend(
                [pmo.constraint(upperBound(c[i])) for i in range(len(c))]
            )
        elif isinstance(c, pmo.variable_dict):
            lowerBoundConstrs.extend([pmo.constraint(lowerBound(c[i])) for i in c])
            upperBoundConstrs.extend([pmo.constraint(upperBound(c[i])) for i in c])
        elif isinstance(c, pmo.variable):
            lowerBoundConstrs.append(pmo.constraint(lowerBound(c)))
            upperBoundConstrs.append(pmo.constraint(upperBound(c)))

        elif isinstance(c, (pmo.objective_list, pmo.objective_tuple)):
            for i in range(len(c)):
                c[i].deactivate()
        elif isinstance(c, pmo.objective_dict):
            for i in c:
                c[i].deactivate()
        elif isinstance(c, pmo.objective):
            c.deactivate()

        elif isinstance(c, (pmo.constraint_list, pmo.constraint_tuple)):
            for i in range(len(c)):
                slackVar, lower, upper = ConfigureConstraint(c[i])
                slackVars.append(slackVar)
                if lower is not None:
                    lowerBoundConstrs.append(lower)
                if upper is not None:
                    upperBoundConstrs.append(upper)
        elif isinstance(c, pmo.constraint_dict):
            for i in c:
                slackVar, lower, upper = ConfigureConstraint(c[i])
                slackVars.append(slackVar)
                if lower is not None:
                    lowerBoundConstrs.append(lower)
                if upper is not None:
                    upperBoundConstrs.append(upper)
        elif isinstance(c, pmo.constraint):
            slackVar, lower, upper = ConfigureConstraint(c)
            slackVars.append(slackVar)
            if lower is not None:
                lowerBoundConstrs.append(lower)
            if upper is not None:
                upperBoundConstrs.append(upper)

    model.slackVars = pmo.variable_list(slackVars)
    model.lowerBoundConstrs = pmo.constraint_list(lowerBoundConstrs)
    model.upperBoundConstrs = pmo.constraint_list(upperBoundConstrs)

    # Step 4, Augment each sub-model
    allSlackVars = deque(model.slackVars)  # type: ignore
    for c in model.children():
        if isinstance(c, (pmo.block_list, pmo.block_tuple)):
            for i in range(len(c)):
                allSlackVars.extend(AugmentModel(c[i]))
        elif isinstance(c, pmo.block_dict):
            for i in c:
                allSlackVars.extend(AugmentModel(c[i]))
        elif isinstance(c, pmo.block):
            allSlackVars.extend(AugmentModel(c))
    return allSlackVars


def CopySolution(fromModel: pmo.block, toModel: pmo.block):
    for toC in toModel.children():
        cName = toC.local_name
        fromC = getattr(fromModel, cName)
        if isinstance(fromC, (pmo.variable_list, pmo.variable_tuple)):
            for i in range(len(fromC)):
                toC[i].value = fromC[i].value
        elif isinstance(fromC, pmo.variable_dict):
            for i in fromC:
                toC[i].value = fromC[i].value
        elif isinstance(fromC, pmo.variable):
            toC.value = fromC.value

        elif isinstance(fromC, (pmo.block_list, pmo.block_tuple)):
            for i in range(len(fromC)):
                CopySolution(fromC[i], toC[i])
        elif isinstance(fromC, pmo.block_dict):
            for i in fromC:
                CopySolution(fromC[i], toC[i])
        elif isinstance(fromC, pmo.block):
            CopySolution(fromC, toC)


def FindLeastInfeasibleSolution(
    originalModel: pmo.block,
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
    originalModel: pmo.block
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

    # Step 1: Augment the model
    augmentedModel = originalModel.clone()
    slackVars = AugmentModel(augmentedModel)

    if leastInfeasibleDefinition == LeastInfeasibleDefinition.L1_Norm:
        augmentedModel.LEAST_INFEASIBLE_L1_OBJ = pmo.objective(
            sum(slackVars), sense=pmo.minimize
        )
    elif leastInfeasibleDefinition == LeastInfeasibleDefinition.L2_Norm:
        augmentedModel.LEAST_INFEASIBLE_L2_OBJ = pmo.objective(
            sum(s**2 for s in slackVars), sense=pmo.minimize
        )
    elif leastInfeasibleDefinition in [
        LeastInfeasibleDefinition.Num_Violated_Constrs,
        LeastInfeasibleDefinition.Sequential,
    ]:
        assert (
            "BigM" in kwargs
        ), 'The Num_Violated_Constrs requires a "BigM" parameter to be passed in.'
        BigM = kwargs["BigM"]
        augmentedModel.slackActive = pmo.variable_list(
            [pmo.variable(domain=pmo.Binary) for i in range(len(slackVars))]
        )

        augmentedModel.slackActive_Definition = pmo.constraint_list([pmo.constraint(slackVars[i] <= BigM * augmentedModel.slackActive[i]) for i in range(len(slackVars))])  # type: ignore

        augmentedModel.LEAST_INFEASIBLE_NUM_VIOLATED_OBJ = pmo.objective(sum(augmentedModel.slackActive[i] for i in range(len(slackVars))), sense=pmo.minimize)  # type: ignore
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
        raise Exception(
            f'Something has gone wrong. The solver terminated with condition "{result.solver.termination_condition}". The problem is likely due to variable bounds being defined the "domain" keyword in the variable definition. Please try using "bounds" keyword there.'
        )

    if leastInfeasibleDefinition == LeastInfeasibleDefinition.Sequential:
        # Fix all slack vars that are not active.
        for i in range(len(slackVars)):
            val = pmo.value(augmentedModel.slackActive[i])
            if val <= 0.5:
                slackVars[i].fix(0)
                augmentedModel.slackActive[i].fix(0)
            else:
                augmentedModel.slackActive[i].fix(1)

        augmentedModel.slackActive_Definition.deactivate()
        augmentedModel.LEAST_INFEASIBLE_NUM_VIOLATED_OBJ.deactivate()

        augmentedModel.LEAST_INFEASIBLE_L1_OBJ = pmo.Objective(
            expr=sum(slackVars), sense=pmo.minimize
        )

        result = solver.solve(augmentedModel, *solver_args, **solver_kwargs)
        if result.solver.termination_condition != TerminationCondition.optimal:
            raise Exception(
                'Something has gone wrong. The problem is likely due to variable bounds being defined the "domain" keyword in the variable definition. Please try using "bounds" keyword there.'
            )

    # Step 6: Copy the solution from the augmented model back to the original model.
    CopySolution(augmentedModel, originalModel)
