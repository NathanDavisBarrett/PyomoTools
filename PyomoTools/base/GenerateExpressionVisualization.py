from .GenerateExpressionString import GenerateExpressionStrings
from .VisualizeExpression import visualize_expression


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
    symStr, numStr = GenerateExpressionStrings(expr)
    visualization = visualize_expression(numStr)

    return f"{symStr}\n{visualization}"
