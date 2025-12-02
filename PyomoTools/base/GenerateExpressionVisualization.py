from .GenerateExpressionString import GenerateExpressionStrings


def GenerateExpressionVisualization(expr):
    """
    A function to generate a visualization of a pyomo expression, showing both the symbolic and numeric forms aligned with connectors.

    Parameters
    ----------
    expr: pyomo expression object
        The expression you'd like to visualize.

    Returns
    -------
    str:
        A string visualization of the expression.
    """
    # Lazy import to avoid circular dependency
    from .ParseNumericExpressionString import visualize_expression

    symStr, numStr = GenerateExpressionStrings(expr)
    visualization = visualize_expression(numStr)

    return f"{symStr}\n{visualization}"
