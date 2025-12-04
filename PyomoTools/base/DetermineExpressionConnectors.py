from typing import List, Tuple


def generate_connectors_and_aligned_expression(
    expression: List[str],
    starts: List[int],
    ends: List[int],
    unchanged_indices: List[int] = None,
    compact: bool = False,
) -> Tuple[str, str, str, List[int], List[int]]:
    """
    Generate the aligned expression and connector lines.

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

    non_operator_indices = [
        i
        for i, token in enumerate(expression)
        if token not in {"+", "-", "*", "/", "^"}
    ]

    # Constraint 1: Connecting lines cannot cross.

    # Constraint 2: Aligned expression lengths must match original token lengths

    # Constraint 3: Bottom connector positions must be within aligned token spans

    # Constraint 4: Aligned tokens need at least one space between them (start and end must be separated by 2)

    # Objective: Minimize total width of aligned expression

    if not compact:
        # Connectors just go stright down
        positions = [(starts[i] + ends[i]) // 2 for i in range(len(expression))]
        top_positions = [positions[ii] for ii in non_operator_indices]
        bottom_positions = top_positions
        token_lengths = [len(expression[ii]) for ii in range(len(expression))]
        aligned_starts = [
            positions[i] - token_lengths[i] // 2 for i in range(len(expression))
        ]
        aligned_ends = [
            aligned_starts[i] + token_lengths[i] - 1 for i in range(len(expression))
        ]
    else:
        # Algorithm:
        # 1. Set the initial aligned position to be as compact as possible (keeping spacing between tokens)
        # 2. Keep track of the right-most connector position observed so far
        # 3. Working from left to right,
        #    Determine the left-most possible way to connect each token from top to bottom.
        #    If this causes an overlap, shift the aligned positions so that the right edge of the aligned token is equal to the maximum observed position in the connector row.
        token_lengths = [len(expression[ii]) for ii in range(len(expression))]
        aligned_starts = []
        aligned_ends = []
        current_position = 0
        for i in range(len(expression)):
            if i > 0:
                current_position += 1  # Minimum spacing between tokens
            aligned_starts.append(current_position)
            aligned_ends.append(current_position + token_lengths[i] - 1)
            current_position += token_lengths[i]

        right_most_connector = -1
        top_positions = []
        bottom_positions = []
        for i in range(len(non_operator_indices)):
            ii = non_operator_indices[i]
            bottom = aligned_ends[ii]
            top = starts[ii]
            if bottom <= ends[ii] and bottom >= starts[ii]:
                top = bottom
            elif bottom < starts[ii]:
                top = starts[ii]
            else:  # bottom > ends[ii]
                top = ends[ii]

            if bottom < right_most_connector + 1:
                required_shift = (right_most_connector + 1) - bottom
                for j in range(ii, len(expression)):
                    aligned_starts[j] += required_shift
                    aligned_ends[j] += required_shift
                bottom = aligned_ends[ii]

            top_positions.append(top)
            bottom_positions.append(bottom)
            right_most_connector = max(right_most_connector, top)

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
