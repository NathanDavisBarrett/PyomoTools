from .Tokenization import Tokenizer
from .Evaluator import Evaluator
from .AST import AST, ASTNode
from .Parser import Parser
from .DetermineLeadingLines import DetermineLeadingLines
from .EmptyExpressionError import EmptyExpressionError

from collections import deque


def visualize_expression(expression: str, plot_ast: bool = False) -> str:
    """
    Visualize the step-by-step evaluation of a mathematical expression.

    This function parses the expression, evaluates it step by step following
    the order of operations (PEMDAS), and generates a visualization showing
    connecting lines between each step.

    Args:
        expression: A mathematical expression string.
                   Supports: +, -, *, /, ^ (power)
                   Functions: sin, cos, tan, log, ln, exp, sqrt, abs, floor, ceil
                   Constants: pi, e
                   Parentheses for grouping
        compact: If True, removes unnecessary whitespace from the output while
                maintaining proper alignment of connector lines. At least one
                space is preserved between tokens.
        balance: If True, balances the AST tree before evaluation to minimize
                depth. This allows more operations to be evaluated in parallel
                for associative operators (+ and *). For example, 1+2+3+4
                evaluated sequentially takes 3 steps, but balanced takes 2 steps.

    Returns:
        A multi-line string showing the step-by-step evaluation with
        connecting lines indicating the flow.
    """
    # Step 1: Tokenize the expression
    tokenizer = Tokenizer(expression)
    tokens = tokenizer.tokenize()

    # Step 2: Parse tokens into an AST
    try:
        ast = Parser(tokens).parse()
    except EmptyExpressionError:
        return "<empty expression>"
    if plot_ast:
        return ast.plot()

    # Step 3: Evaluate the AST step-by-step
    evaluator = Evaluator(ast, expression).iterates

    # Step 4: Generate leading lines for each iterate (and collect output)
    output_lines = deque()
    for i in range(len(evaluator) - 1):
        current_iterate = evaluator[i]
        output_lines.append(current_iterate.expression_str)

        startGroupings = list(current_iterate.start_groupings)
        endGroupings = list(current_iterate.end_groupings)

        leadingLines = DetermineLeadingLines(startGroupings, endGroupings)

        output_lines.extend(leadingLines)

    # Append the final expression
    output_lines.append(evaluator[-1].expression_str)
    return "\n".join(output_lines)
