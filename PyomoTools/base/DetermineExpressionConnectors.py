from typing import List, Tuple

import pyomo.kernel as pmo

# from .Solvers.DefaultSolver import DefaultSolver
print("Use real function")
DefaultSolver = lambda _: pmo.SolverFactory("gurobi")


def _value(var):
    """Extract integer value from a Pyomo variable."""
    val = pmo.value(var, exception=False)
    if val is None:
        return None
    return int(round(val))


def generate_connectors_and_aligned_expression(
    expression: List[str],
    starts: List[int],
    ends: List[int],
    unchanged_indices: List[int] = None,
    compact: bool = False,
) -> Tuple[str, str, str, List[int], List[int]]:
    """
    Generate the aligned expression and connector lines using integer programming.

    This function solves an optimization problem to determine:
    1. Where connectors should attach to the old expression (top positions)
    2. Where connectors should attach to the new expression (bottom positions)
    3. Where each token should be placed in the aligned expression

    Example:
    ~~~~~~~~~
    Preliminary data (old expression):
    3 + 5 * 2
    ~~~~~~~~~
    Generated connectors and positions:

    │   ├───┘ (Top connector line - brackets under evaluated expressions)
    │   │     (Bottom connector line - vertical drops to new positions)
    3 + 10    (Aligned expression)

    expression: ['3', '+', '10']
    starts: [0, 2, 4]
    ends:   [0, 2, 8] (8 since this is the ending position of the previous row's '5 * 2' string)
    unchanged_indices: [0] (index 0, token '3', is unchanged - just passing through)

    Args:
        expression (List[str]): List of strings representing each token of the new expression.
        starts (List[int]): List of starting positions for each token in the OLD expression.
        ends (List[int]): List of ending positions for each token in the OLD expression.
        unchanged_indices (List[int]): Indices of tokens that are unchanged (just passing through).
            These get simple │ connectors instead of bracket connectors.
        compact (bool): Whether to generate a compact representation. If False, connectors
                       can only go straight down. If True, connectors can angle to minimize width.

    Returns:
        Tuple containing:
        - connector_line_1 (str): The top connector line with brackets (└, ─, ┘, │, etc.)
        - connector_line_2 (str): The bottom connector line showing vertical/angled connections
        - aligned_expression (str): The new expression string with proper spacing
        - aligned_starts (List[int]): Starting positions of each token in the aligned expression
        - aligned_ends (List[int]): Ending positions of each token in the aligned expression
    """
    if unchanged_indices is None:
        unchanged_indices = []
    unchanged_set = set(unchanged_indices)

    model = pmo.block()
    non_operator_indices = [
        i
        for i, token in enumerate(expression)
        if token not in {"+", "-", "*", "/", "^"}
    ]

    model.top_pos = pmo.variable_list(
        [
            pmo.variable(domain=pmo.Integers, lb=starts[i], ub=ends[i])
            for i in non_operator_indices
        ]
    )  # Bounds handle the fact that top connectors must be within the token span
    model.bottom_pos = pmo.variable_list(
        [pmo.variable(domain=pmo.NonNegativeIntegers) for _ in non_operator_indices]
    )
    model.aligned_start = pmo.variable_list(
        [pmo.variable(domain=pmo.NonNegativeIntegers) for _ in expression]
    )
    model.aligned_end = pmo.variable_list(
        [pmo.variable(domain=pmo.NonNegativeIntegers) for _ in expression]
    )

    # Constraint 1: Connecting lines cannot cross.
    # # Mathematically, this means that
    #   bottom_pos[i+1] >= bottom_pos[i] + 1
    #   bottom_pos[i+1] >= top_pos[i] + 1
    model.no_overlap_1 = pmo.constraint_list()
    model.no_overlap_2 = pmo.constraint_list()
    for idx in range(len(non_operator_indices) - 1):
        model.no_overlap_1.append(
            pmo.constraint(model.bottom_pos[idx + 1] >= model.bottom_pos[idx] + 1)
        )
        model.no_overlap_2.append(
            pmo.constraint(model.bottom_pos[idx + 1] >= model.top_pos[idx] + 1)
        )

    # Constraint 2: Aligned expression lengths must match original token lengths
    model.length_match = pmo.constraint_list(
        [
            pmo.constraint(
                model.aligned_end[i] - model.aligned_start[i] == len(expression[i]) - 1
            )
            for i in range(len(expression))
        ]
    )

    # Constraint 3: Bottom connector positions must be within aligned token spans
    model.bottom_within_span_1 = pmo.constraint_list(
        [
            pmo.constraint(
                model.bottom_pos[i] >= model.aligned_start[non_operator_indices[i]]
            )
            for i in range(len(non_operator_indices))
        ]
    )
    model.bottom_within_span_2 = pmo.constraint_list(
        [
            pmo.constraint(
                model.bottom_pos[i] <= model.aligned_end[non_operator_indices[i]]
            )
            for i in range(len(non_operator_indices))
        ]
    )

    # Constraint 4: Aligned tokens need at least one space between them (start and end must be separated by 2)
    model.token_spacing = pmo.constraint_list(
        [
            pmo.constraint(model.aligned_start[i + 1] >= model.aligned_end[i] + 2)
            for i in range(len(expression) - 1)
        ]
    )

    if not compact:
        # Enforce that connectors can only go straight down
        model.straight_connectors = pmo.constraint_list(
            [
                pmo.constraint(model.top_pos[i] == model.bottom_pos[i])
                for i in range(len(non_operator_indices))
            ]
        )

    # Objective: Minimize total width of aligned expression
    model.obj1 = pmo.objective(expr=model.aligned_end[-1], sense=pmo.minimize)

    solver = DefaultSolver("MILP")

    results = solver.solve(model, tee=False)
    if results.solver.termination_condition != pmo.TerminationCondition.optimal:
        raise Exception("Failed to solve alignment integer program optimally.")

    # Objective: Center connectors within token spans
    min_width = pmo.value(model.obj1)
    model.obj1.deactivate()
    model.aligned_end[-1].fix(min_width)

    model.top_center = pmo.variable_list(
        [
            pmo.variable(
                domain=pmo.Reals,
                value=(
                    (starts[non_operator_indices[i]] + ends[non_operator_indices[i]])
                    / 2
                ),
            )
            for i in range(len(non_operator_indices))
        ]
    )
    model.bot_center = pmo.variable_list(
        [
            pmo.variable(
                domain=pmo.Reals,
                value=pmo.value(
                    model.aligned_start[non_operator_indices[i]]
                    + model.aligned_end[non_operator_indices[i]]
                )
                / 2,
            )
            for i in range(len(non_operator_indices))
        ]
    )

    model.top_center_def = pmo.constraint_list(
        [
            pmo.constraint(
                model.top_center[i]
                == (starts[non_operator_indices[i]] + ends[non_operator_indices[i]]) / 2
            )
            for i in range(len(non_operator_indices))
        ]
    )
    model.bot_center_def = pmo.constraint_list(
        [
            pmo.constraint(
                model.bot_center[i]
                == (
                    model.aligned_start[non_operator_indices[i]]
                    + model.aligned_end[non_operator_indices[i]]
                )
                / 2
            )
            for i in range(len(non_operator_indices))
        ]
    )

    model.centering_obj = pmo.objective(
        expr=sum(
            (model.top_center[i] - model.top_pos[i]) ** 2
            + (model.bot_center[i] - model.bottom_pos[i]) ** 2
            for i in range(len(non_operator_indices))
        ),
        sense=pmo.minimize,
    )

    results = solver.solve(model, tee=False, warmstart=True, options={"TimeLimit": 1})
    if results.solver.termination_condition != pmo.TerminationCondition.optimal:
        raise Exception("Failed to solve alignment integer program optimally.")

    # Extract variable values
    top_positions = [_value(model.top_pos[i]) for i in range(len(non_operator_indices))]
    for i in range(len(top_positions)):
        if top_positions[i] is None:
            ii = non_operator_indices[i]
            top_positions[i] = (starts[ii] + ends[ii]) // 2
    bottom_positions = [
        _value(model.bottom_pos[i]) for i in range(len(non_operator_indices))
    ]
    aligned_starts = [_value(model.aligned_start[i]) for i in range(len(expression))]
    aligned_ends = [_value(model.aligned_end[i]) for i in range(len(expression))]

    max_top = max(ends)

    connector_line_1 = [" " for _ in range(max_top + 1)]
    for i in range(len(non_operator_indices)):
        ii = non_operator_indices[i]

        # Check if this token is unchanged (just passing through)
        if ii in unchanged_set:
            # Unchanged token - just draw a vertical line at top position
            connector_line_1[top_positions[i]] = "│"
            continue

        num_left = top_positions[i] - starts[ii]
        num_right = ends[ii] - top_positions[i]

        # Check if there are any unchanged tokens to the left that need pass-through
        has_passthrough_left = any(
            j in unchanged_set and starts[j] < starts[ii]
            for j in range(len(expression))
            if j not in {"+", "-", "*", "/", "^"}
            and expression[j] not in {"+", "-", "*", "/", "^"}
        )

        if num_left == 0 and num_right == 0:
            connector_line_1[top_positions[i]] = "│"
        elif num_left == 0:
            # Use ├ if there's a passthrough to the left, otherwise └
            if has_passthrough_left:
                connector_line_1[top_positions[i]] = "├"
            else:
                connector_line_1[top_positions[i]] = "├"  # "└"
            for j in range(1, num_right):
                connector_line_1[top_positions[i] + j] = "─"
            connector_line_1[ends[ii]] = "┘"
        elif num_right == 0:
            for j in range(num_left - 1):
                connector_line_1[top_positions[i] - j - 1] = "─"
            connector_line_1[top_positions[i]] = "┤"
            connector_line_1[starts[ii]] = "└"
        else:
            for j in range(num_left - 1):
                connector_line_1[top_positions[i] - j - 1] = "─"
            for j in range(1, num_right):
                connector_line_1[top_positions[i] + j] = "─"

            connector_line_1[top_positions[i]] = "┬"
            connector_line_1[starts[ii]] = "└"
            connector_line_1[ends[ii]] = "┘"

    connector_line_1_str = "".join(connector_line_1)

    max_bottom = max(max(bottom_positions), max(top_positions))
    connector_line_2 = [" " for _ in range(max_bottom + 1)]

    for i in range(len(non_operator_indices)):
        top = top_positions[i]
        bottom = bottom_positions[i]

        if top == bottom:
            connector_line_2[bottom] = "│"
        elif top < bottom:
            connector_line_2[bottom] = "┐"
            connector_line_2[top] = "└"
            for j in range(1, bottom - top):
                connector_line_2[top + j] = "─"
        else:
            connector_line_2[bottom] = "┌"
            connector_line_2[top] = "┘"
            for j in range(1, top - bottom):
                connector_line_2[bottom + j] = "─"

    connector_line_2_str = "".join(connector_line_2)

    max_allgined = max(aligned_ends)
    aligned_expression_parts = [" " for _ in range(max_allgined + 1)]
    for i in range(len(expression)):
        start = aligned_starts[i]
        for j, char in enumerate(expression[i]):
            aligned_expression_parts[start + j] = char

    aligned_expression_str = "".join(aligned_expression_parts)

    return (
        connector_line_1_str,
        connector_line_2_str,
        aligned_expression_str,
        aligned_starts,
        aligned_ends,
    )


if __name__ == "__main__":
    original_expression = "3 + 5 * (2 - 8) / 4 ^ 2"

    expression = ["3", "+", "5", "*", "-6", "/", "16"]
    starts = [0, 2, 4, 6, 8, 16, 18]
    ends = [0, 2, 4, 6, 14, 16, 22]

    conn1, conn2, aligned_expr, aligned_starts, aligned_ends = (
        generate_connectors_and_aligned_expression(
            expression, starts, ends, compact=True
        )
    )
    print(original_expression)
    print(conn1)
    print(conn2)
    print(aligned_expr)
    print(f"Aligned starts: {aligned_starts}")
    print(f"Aligned ends: {aligned_ends}")
