"""
Module for visualizing step-by-step evaluation of numeric expressions.

Supports common mathematical operators (+, -, *, /, ^), functions (sin, cos, log, exp, sqrt),
and parentheses for grouping. Displays evaluation with connecting lines showing the flow.

Example:
    >>> print(visualize_expression("3 + 5 * (2 - 8) / 4 ^ 2"))
    3 + 5 * (2 - 8) / 4 ^ 2
    │   │   └──┬──┘   └─┬─┘
    │   │      │   ┌────┘
    3 + 5 *   -6 / 16
    │   └───┬──┘   │
    │       │   ┌──┘
    3 +   -30 / 16
    │     └───┬──┘
    │         │
    3 +   -1.875
    └────┬─────┘
         │
      1.125
"""

import math
from typing import List, Tuple, Optional, Dict

print("MAKE RELATIVE!")
from DetermineExpressionConnectors import generate_connectors_and_aligned_expression
from dataclasses import dataclass, field
from enum import Enum, auto


# =============================================================================
# Token Types and Tokenizer
# =============================================================================


class TokenType(Enum):
    NUMBER = auto()
    OPERATOR = auto()
    LPAREN = auto()
    RPAREN = auto()
    FUNCTION = auto()
    COMMA = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    start: int  # Start position in original string
    end: int  # End position in original string


class Tokenizer:
    """Tokenize a mathematical expression string."""

    FUNCTIONS = {
        "sin",
        "cos",
        "tan",
        "log",
        "ln",
        "exp",
        "sqrt",
        "abs",
        "floor",
        "ceil",
        "asin",
        "acos",
        "atan",
    }
    OPERATORS = {"+", "-", "*", "/", "^"}
    CONSTANTS = {"pi": math.pi, "e": math.e}

    def __init__(self, expression: str):
        self.expression = expression
        self.pos = 0
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """Convert expression string to list of tokens."""
        while self.pos < len(self.expression):
            self._skip_whitespace()
            if self.pos >= len(self.expression):
                break

            char = self.expression[self.pos]

            if char.isdigit() or char == ".":
                self._read_number()
            elif char.isalpha() or char == "_":
                self._read_identifier()
            elif char in self.OPERATORS:
                self.tokens.append(
                    Token(TokenType.OPERATOR, char, self.pos, self.pos + 1)
                )
                self.pos += 1
            elif char == "(":
                self.tokens.append(
                    Token(TokenType.LPAREN, char, self.pos, self.pos + 1)
                )
                self.pos += 1
            elif char == ")":
                self.tokens.append(
                    Token(TokenType.RPAREN, char, self.pos, self.pos + 1)
                )
                self.pos += 1
            elif char == ",":
                self.tokens.append(Token(TokenType.COMMA, char, self.pos, self.pos + 1))
                self.pos += 1
            else:
                raise ValueError(
                    f"Unexpected character '{char}' at position {self.pos}"
                )

        return self.tokens

    def _skip_whitespace(self):
        while self.pos < len(self.expression) and self.expression[self.pos].isspace():
            self.pos += 1

    def _read_number(self):
        start = self.pos
        has_dot = False
        has_e = False

        while self.pos < len(self.expression):
            char = self.expression[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == "." and not has_dot and not has_e:
                has_dot = True
                self.pos += 1
            elif char.lower() == "e" and not has_e:
                has_e = True
                self.pos += 1
                if (
                    self.pos < len(self.expression)
                    and self.expression[self.pos] in "+-"
                ):
                    self.pos += 1
            else:
                break

        self.tokens.append(
            Token(TokenType.NUMBER, self.expression[start : self.pos], start, self.pos)
        )

    def _read_identifier(self):
        start = self.pos
        while self.pos < len(self.expression) and (
            self.expression[self.pos].isalnum() or self.expression[self.pos] == "_"
        ):
            self.pos += 1

        name = self.expression[start : self.pos]
        name_lower = name.lower()

        if name_lower in self.FUNCTIONS:
            self.tokens.append(Token(TokenType.FUNCTION, name_lower, start, self.pos))
        elif name_lower in self.CONSTANTS:
            self.tokens.append(
                Token(
                    TokenType.NUMBER, str(self.CONSTANTS[name_lower]), start, self.pos
                )
            )
        else:
            raise ValueError(f"Unknown identifier '{name}' at position {start}")


# =============================================================================
# AST Node Types
# =============================================================================


@dataclass
class ASTNode:
    """Base class for AST nodes with position tracking."""

    start_pos: int = 0
    end_pos: int = 0


@dataclass
class NumberNode(ASTNode):
    value: float = 0.0
    original_str: str = ""


@dataclass
class BinaryOpNode(ASTNode):
    op: str = ""
    left: ASTNode = None
    right: ASTNode = None


@dataclass
class UnaryOpNode(ASTNode):
    op: str = ""
    operand: ASTNode = None


@dataclass
class FunctionNode(ASTNode):
    name: str = ""
    args: List[ASTNode] = field(default_factory=list)


# =============================================================================
# Parser
# =============================================================================


class Parser:
    """Parse tokens into an AST using recursive descent with operator precedence."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> ASTNode:
        """Parse tokens into AST."""
        if not self.tokens:
            raise ValueError("Empty expression")
        result = self._parse_expression()
        if self.pos < len(self.tokens):
            raise ValueError(
                f"Unexpected token at position {self.tokens[self.pos].start}: {self.tokens[self.pos].value}"
            )
        return result

    def _current(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(
        self, expected_type: TokenType = None, expected_value: str = None
    ) -> Token:
        token = self._current()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if expected_type and token.type != expected_type:
            raise ValueError(
                f"Expected {expected_type}, got {token.type} at position {token.start}"
            )
        if expected_value and token.value != expected_value:
            raise ValueError(
                f"Expected '{expected_value}', got '{token.value}' at position {token.start}"
            )
        self.pos += 1
        return token

    def _parse_expression(self) -> ASTNode:
        return self._parse_additive()

    def _parse_additive(self) -> ASTNode:
        left = self._parse_multiplicative()

        while (
            self._current()
            and self._current().type == TokenType.OPERATOR
            and self._current().value in "+-"
        ):
            op_token = self._consume()
            right = self._parse_multiplicative()
            left = BinaryOpNode(
                start_pos=left.start_pos,
                end_pos=right.end_pos,
                op=op_token.value,
                left=left,
                right=right,
            )

        return left

    def _parse_multiplicative(self) -> ASTNode:
        left = self._parse_power()

        while (
            self._current()
            and self._current().type == TokenType.OPERATOR
            and self._current().value in "*/"
        ):
            op_token = self._consume()
            right = self._parse_power()
            left = BinaryOpNode(
                start_pos=left.start_pos,
                end_pos=right.end_pos,
                op=op_token.value,
                left=left,
                right=right,
            )

        return left

    def _parse_power(self) -> ASTNode:
        left = self._parse_unary()

        # Right-associative
        if (
            self._current()
            and self._current().type == TokenType.OPERATOR
            and self._current().value == "^"
        ):
            op_token = self._consume()
            right = self._parse_power()  # Right-associative recursion
            left = BinaryOpNode(
                start_pos=left.start_pos,
                end_pos=right.end_pos,
                op=op_token.value,
                left=left,
                right=right,
            )

        return left

    def _parse_unary(self) -> ASTNode:
        if (
            self._current()
            and self._current().type == TokenType.OPERATOR
            and self._current().value in "+-"
        ):
            op_token = self._consume()
            operand = self._parse_unary()
            return UnaryOpNode(
                start_pos=op_token.start,
                end_pos=operand.end_pos,
                op=op_token.value,
                operand=operand,
            )

        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        token = self._current()

        if token is None:
            raise ValueError("Unexpected end of expression")

        if token.type == TokenType.NUMBER:
            self._consume()
            return NumberNode(
                start_pos=token.start,
                end_pos=token.end,
                value=float(token.value),
                original_str=token.value,
            )

        if token.type == TokenType.FUNCTION:
            return self._parse_function()

        if token.type == TokenType.LPAREN:
            lparen = self._consume()
            expr = self._parse_expression()
            rparen = self._consume(TokenType.RPAREN)
            expr.start_pos = lparen.start
            expr.end_pos = rparen.end
            return expr

        raise ValueError(f"Unexpected token '{token.value}' at position {token.start}")

    def _parse_function(self) -> ASTNode:
        func_token = self._consume(TokenType.FUNCTION)
        self._consume(TokenType.LPAREN)  # consume opening paren

        args = []
        if self._current() and self._current().type != TokenType.RPAREN:
            args.append(self._parse_expression())
            while self._current() and self._current().type == TokenType.COMMA:
                self._consume()
                args.append(self._parse_expression())

        rparen = self._consume(TokenType.RPAREN)

        return FunctionNode(
            start_pos=func_token.start,
            end_pos=rparen.end,
            name=func_token.value,
            args=args,
        )


# =============================================================================
# Evaluator
# =============================================================================


class Evaluator:
    """Evaluate AST nodes."""

    FUNCTIONS = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log10,
        "ln": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "abs": abs,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    def evaluate(self, node: ASTNode) -> float:
        """Evaluate an AST node and return the result."""
        if isinstance(node, NumberNode):
            return node.value

        if isinstance(node, UnaryOpNode):
            operand = self.evaluate(node.operand)
            if node.op == "-":
                return -operand
            return operand

        if isinstance(node, BinaryOpNode):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)

            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            if node.op == "^":
                return left**right

        if isinstance(node, FunctionNode):
            args = [self.evaluate(arg) for arg in node.args]
            func = self.FUNCTIONS.get(node.name)
            if func:
                return func(*args)
            raise ValueError(f"Unknown function: {node.name}")

        raise ValueError(f"Unknown node type: {type(node)}")


# =============================================================================
# Expression String Builder
# =============================================================================


def format_number(value: float, precision: int = 10) -> str:
    """Format a number for display."""
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    formatted = f"{value:.{precision}g}"
    return formatted


def get_precedence(op: str) -> int:
    """Get operator precedence (higher = binds tighter)."""
    if op in "+-":
        return 1
    if op in "*/":
        return 2
    if op == "^":
        return 3
    return 0


def is_right_associative(op: str) -> bool:
    """Check if operator is right-associative."""
    return op == "^"


@dataclass
class ExpressionSegment:
    """A segment of an expression string with metadata."""

    text: str
    node_id: int = -1  # ID of the AST node this segment represents
    is_operator: bool = False
    is_paren: bool = False


class ExpressionBuilder:
    """Build expression string from AST with position tracking."""

    def __init__(self):
        self.node_positions: Dict[int, Tuple[int, int]] = {}  # node_id -> (start, end)

    def build(self, node: ASTNode, node_id_map: Dict[int, ASTNode]) -> str:
        """Build expression string and track positions of nodes."""
        self.node_positions = {}
        result = self._build_node(node, 0, False, node_id_map)
        return result

    def _build_node(
        self,
        node: ASTNode,
        parent_prec: int,
        is_right: bool,
        node_id_map: Dict[int, ASTNode],
        offset: int = 0,
    ) -> str:
        """Build string for a node, tracking positions."""
        node_id = id(node)

        if isinstance(node, NumberNode):
            text = node.original_str if node.original_str else format_number(node.value)
            self.node_positions[node_id] = (offset, offset + len(text))
            return text

        if isinstance(node, UnaryOpNode):
            operand_str = self._build_node(
                node.operand, 4, False, node_id_map, offset + 1
            )
            text = f"{node.op}{operand_str}"
            self.node_positions[node_id] = (offset, offset + len(text))
            return text

        if isinstance(node, BinaryOpNode):
            prec = get_precedence(node.op)
            need_parens = prec < parent_prec or (
                prec == parent_prec and is_right and not is_right_associative(node.op)
            )

            current_offset = offset + (1 if need_parens else 0)

            left_str = self._build_node(
                node.left, prec, False, node_id_map, current_offset
            )
            op_str = f" {node.op} "
            right_offset = current_offset + len(left_str) + len(op_str)
            right_str = self._build_node(
                node.right, prec, True, node_id_map, right_offset
            )

            inner = f"{left_str}{op_str}{right_str}"
            if need_parens:
                text = f"({inner})"
            else:
                text = inner

            self.node_positions[node_id] = (offset, offset + len(text))
            return text

        if isinstance(node, FunctionNode):
            args_strs = []
            current_offset = offset + len(node.name) + 1  # name + (
            for i, arg in enumerate(node.args):
                if i > 0:
                    current_offset += 2  # ", "
                arg_str = self._build_node(arg, 0, False, node_id_map, current_offset)
                args_strs.append(arg_str)
                current_offset += len(arg_str)

            text = f"{node.name}({', '.join(args_strs)})"
            self.node_positions[node_id] = (offset, offset + len(text))
            return text

        return ""


# =============================================================================
# Step-by-Step Evaluation Tracker
# =============================================================================


@dataclass
class EvaluationStep:
    """Represents one step in the evaluation."""

    expression: str
    evaluated_positions: List[Tuple[int, int]]  # (start, end) of evaluated parts
    results: List[str]  # Result strings
    result_positions: List[int]  # Center positions of results in next expression


def is_effectively_number(node: ASTNode) -> bool:
    """
    Check if a node is effectively a number - either a NumberNode directly,
    or a UnaryOpNode applied to a NumberNode (like -5 or +3).
    """
    if isinstance(node, NumberNode):
        return True
    if isinstance(node, UnaryOpNode) and isinstance(node.operand, NumberNode):
        return True
    return False


def find_evaluatable_nodes(node: ASTNode) -> List[ASTNode]:
    """
    Find all nodes that can be evaluated in the next step.
    These are nodes where all children are effectively numbers
    (NumberNodes or UnaryOpNodes of NumberNodes).
    """
    result = []

    if isinstance(node, NumberNode):
        return []

    if isinstance(node, UnaryOpNode):
        if isinstance(node.operand, NumberNode):
            result.append(node)
        else:
            result.extend(find_evaluatable_nodes(node.operand))

    elif isinstance(node, BinaryOpNode):
        left_ready = is_effectively_number(node.left)
        right_ready = is_effectively_number(node.right)

        if left_ready and right_ready:
            result.append(node)
        else:
            if not left_ready:
                result.extend(find_evaluatable_nodes(node.left))
            if not right_ready:
                result.extend(find_evaluatable_nodes(node.right))

    elif isinstance(node, FunctionNode):
        all_args_ready = all(is_effectively_number(arg) for arg in node.args)
        if all_args_ready:
            result.append(node)
        else:
            for arg in node.args:
                if not is_effectively_number(arg):
                    result.extend(find_evaluatable_nodes(arg))

    return result

    return result


def find_highest_precedence_nodes(evaluatable: List[ASTNode]) -> List[ASTNode]:
    """
    From evaluatable nodes, find all nodes that can be evaluated in parallel.

    The strategy is to evaluate all independent "ready" nodes at the same time.
    A node is "ready" if all its children are numbers.
    Nodes are independent if they don't overlap in the expression tree.

    We return all evaluatable nodes since they're all ready by definition,
    and the find_evaluatable_nodes function already ensures they don't overlap.
    """
    if not evaluatable:
        return []

    # All evaluatable nodes are ready and independent by construction
    return evaluatable


def evaluate_and_replace(
    root: ASTNode, to_evaluate: List[ASTNode], evaluator: Evaluator
) -> ASTNode:
    """
    Create a new AST with specified nodes replaced by their evaluated values.
    """
    eval_ids = {id(n) for n in to_evaluate}

    def transform(node: ASTNode) -> ASTNode:
        if id(node) in eval_ids:
            result = evaluator.evaluate(node)
            return NumberNode(
                start_pos=node.start_pos,
                end_pos=node.end_pos,
                value=result,
                original_str=format_number(result),
            )

        if isinstance(node, NumberNode):
            return node

        if isinstance(node, UnaryOpNode):
            return UnaryOpNode(
                start_pos=node.start_pos,
                end_pos=node.end_pos,
                op=node.op,
                operand=transform(node.operand),
            )

        if isinstance(node, BinaryOpNode):
            return BinaryOpNode(
                start_pos=node.start_pos,
                end_pos=node.end_pos,
                op=node.op,
                left=transform(node.left),
                right=transform(node.right),
            )

        if isinstance(node, FunctionNode):
            return FunctionNode(
                start_pos=node.start_pos,
                end_pos=node.end_pos,
                name=node.name,
                args=[transform(arg) for arg in node.args],
            )

        return node

    return transform(root)


# =============================================================================
# Visualization
# =============================================================================


class ExpressionVisualizer:
    """
    Visualize step-by-step evaluation of a mathematical expression.

    Uses integer programming to determine optimal connector positions and
    expression alignment for compact output.
    """

    def __init__(self, expression: str, compact: bool = False):
        self.original_expression = expression
        self.compact = compact
        self.tokenizer = Tokenizer(expression)
        self.tokens = self.tokenizer.tokenize()
        self.parser = Parser(self.tokens)
        self.ast = self.parser.parse()
        self.evaluator = Evaluator()
        self.builder = ExpressionBuilder()

    def visualize(self) -> str:
        """Generate the complete visualization using integer programming for alignment."""
        all_lines = []  # List of output lines
        current_ast = self.ast
        node_id_map = {}
        first_iteration = True

        # Build expression string (for node position tracking)
        # We'll use the original expression on the first iteration to preserve whitespace
        self.builder.build(current_ast, node_id_map)
        current_expr_str = (
            self.original_expression
        )  # Use original with whitespace preserved

        while True:
            # Find nodes to evaluate
            evaluatable = find_evaluatable_nodes(current_ast)
            if not evaluatable:
                all_lines.append(current_expr_str.rstrip())
                break

            to_evaluate = find_highest_precedence_nodes(evaluatable)
            if not to_evaluate:
                all_lines.append(current_expr_str.rstrip())
                break

            # Get positions and results of nodes being evaluated
            eval_positions = []
            for node in to_evaluate:
                if first_iteration:
                    # On first iteration, use AST's original positions
                    start, end = node.start_pos, node.end_pos
                    result = self.evaluator.evaluate(node)
                    eval_positions.append((start, end, format_number(result)))
                else:
                    # On subsequent iterations, use builder's positions
                    node_id = id(node)
                    if node_id in self.builder.node_positions:
                        start, end = self.builder.node_positions[node_id]
                        result = self.evaluator.evaluate(node)
                        eval_positions.append((start, end, format_number(result)))

            # Sort by position
            eval_positions.sort(key=lambda x: x[0])

            # Build the new expression tokens and their position mappings
            new_tokens, token_starts, token_ends, unchanged_indices = (
                self._build_new_expression_tokens(current_expr_str, eval_positions)
            )

            # Use integer programming to generate connectors and aligned expression
            try:
                conn1, conn2, aligned_expr, aligned_starts, aligned_ends = (
                    generate_connectors_and_aligned_expression(
                        new_tokens,
                        token_starts,
                        token_ends,
                        unchanged_indices=unchanged_indices,
                        compact=self.compact,
                    )
                )
            except Exception:
                # If IP fails, fall back to simple alignment
                aligned_expr = " ".join(new_tokens)
                conn1 = ""
                conn2 = ""
                aligned_starts = None
                aligned_ends = None

            # Add current expression and connectors to output
            all_lines.append(current_expr_str.rstrip())
            if conn1.strip():
                all_lines.append(conn1.rstrip())
            if conn2.strip():
                all_lines.append(conn2.rstrip())

            # Evaluate and get new AST
            new_ast = evaluate_and_replace(current_ast, to_evaluate, self.evaluator)

            # Update for next iteration
            current_ast = new_ast
            current_expr_str = aligned_expr
            first_iteration = False  # Use builder positions after first iteration

            # Update node positions based on the aligned expression
            if aligned_starts is not None and aligned_ends is not None:
                self._update_node_positions_from_aligned_positions(
                    new_tokens, aligned_starts, aligned_ends, new_ast
                )
            else:
                self._update_node_positions_from_aligned(
                    aligned_expr, new_ast, node_id_map
                )

            # Check if we're done
            if isinstance(current_ast, NumberNode):
                all_lines.append(aligned_expr.rstrip())
                break

        return "\n".join(all_lines)

    def _build_new_expression_tokens(
        self,
        old_expr: str,
        eval_positions: List[Tuple[int, int, str]],
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        """
        Build the list of tokens for the new expression, along with their
        start/end positions from the old expression.

        Args:
            old_expr: The current expression string
            eval_positions: List of (start, end, result) for evaluated subexpressions

        Returns:
            Tuple of:
            - tokens: List of token strings for the new expression
            - starts: Starting positions of each token in the old expression
            - ends: Ending positions of each token in the old expression
            - unchanged_indices: Indices of tokens that are unchanged (not evaluated)
        """
        tokens = []
        starts = []
        ends = []
        unchanged_indices = []

        # Create a set of positions that are being evaluated
        eval_ranges = []
        for start, end, result in eval_positions:
            eval_ranges.append((start, end, result))

        # Sort eval_ranges by start position
        eval_ranges.sort(key=lambda x: x[0])

        pos = 0
        eval_idx = 0

        while pos < len(old_expr):
            # Skip whitespace
            if old_expr[pos] == " ":
                pos += 1
                continue

            # Check if we're at an evaluated range
            if eval_idx < len(eval_ranges) and pos == eval_ranges[eval_idx][0]:
                start, end, result = eval_ranges[eval_idx]
                tokens.append(result)
                starts.append(start)
                ends.append(end - 1)  # Convert to inclusive end
                # This is an evaluated token, not unchanged
                pos = end
                eval_idx += 1
                continue

            # Check if we're inside an evaluated range (shouldn't happen if sorted correctly)
            in_eval_range = False
            for start, end, _ in eval_ranges:
                if start <= pos < end:
                    in_eval_range = True
                    pos = end
                    break
            if in_eval_range:
                continue

            # Read a token (operator or operand)
            if old_expr[pos] in "+-*/^":
                tokens.append(old_expr[pos])
                starts.append(pos)
                ends.append(pos)
                pos += 1
            elif (
                old_expr[pos].isdigit()
                or old_expr[pos] == "."
                or (
                    old_expr[pos] == "-"
                    and pos + 1 < len(old_expr)
                    and (old_expr[pos + 1].isdigit() or old_expr[pos + 1] == ".")
                    and (pos == 0 or old_expr[pos - 1] in " (+-*/^")
                )
            ):
                # Read a number (including negative numbers at start or after operators)
                start_pos = pos
                if old_expr[pos] == "-":
                    pos += 1
                while pos < len(old_expr) and (
                    old_expr[pos].isdigit() or old_expr[pos] == "."
                ):
                    pos += 1
                token = old_expr[start_pos:pos]
                tokens.append(token)
                starts.append(start_pos)
                ends.append(pos - 1)  # Inclusive end
                # This is an unchanged token
                unchanged_indices.append(len(tokens) - 1)
            else:
                # Skip other characters (parentheses already handled by evaluation)
                pos += 1

        return tokens, starts, ends, unchanged_indices

    def _update_node_positions_from_aligned_positions(
        self,
        tokens: List[str],
        aligned_starts: List[int],
        aligned_ends: List[int],
        ast: ASTNode,
    ) -> None:
        """
        Update builder node positions using the known aligned positions from IP solution.
        """
        self.builder.node_positions = {}

        # Build a map from token index to position
        token_positions = {}
        for i, (start, end) in enumerate(zip(aligned_starts, aligned_ends)):
            token_positions[i] = (start, end + 1)  # Convert back to exclusive end

        # Find node positions based on token indices
        self._map_ast_to_positions(ast, tokens, token_positions, 0)

    def _map_ast_to_positions(
        self,
        node: ASTNode,
        tokens: List[str],
        token_positions: Dict[int, Tuple[int, int]],
        token_idx: int,
    ) -> int:
        """
        Map AST nodes to their positions in the aligned expression.
        Returns the next token index to process.
        """
        node_id = id(node)

        if isinstance(node, NumberNode):
            if token_idx < len(tokens):
                start, end = token_positions.get(token_idx, (0, 1))
                self.builder.node_positions[node_id] = (start, end)
                return token_idx + 1
            return token_idx

        if isinstance(node, BinaryOpNode):
            # Process left child
            left_start_idx = token_idx
            next_idx = self._map_ast_to_positions(
                node.left, tokens, token_positions, token_idx
            )

            # Skip operator
            next_idx += 1

            # Process right child
            right_start_idx = next_idx
            next_idx = self._map_ast_to_positions(
                node.right, tokens, token_positions, next_idx
            )

            # This node spans from left to right
            if left_start_idx < len(tokens) and right_start_idx < len(tokens):
                left_start = token_positions.get(left_start_idx, (0, 1))[0]
                right_end = token_positions.get(next_idx - 1, (0, 1))[1]
                self.builder.node_positions[node_id] = (left_start, right_end)

            return next_idx

        if isinstance(node, UnaryOpNode):
            child_idx = self._map_ast_to_positions(
                node.operand, tokens, token_positions, token_idx
            )
            if token_idx < len(tokens):
                start = token_positions.get(token_idx, (0, 1))[0]
                end = token_positions.get(child_idx - 1, (0, 1))[1]
                self.builder.node_positions[node_id] = (start, end)
            return child_idx

        if isinstance(node, FunctionNode):
            # Functions are replaced entirely, so just use current position
            if token_idx < len(tokens):
                start, end = token_positions.get(token_idx, (0, 1))
                self.builder.node_positions[node_id] = (start, end)
                return token_idx + 1
            return token_idx

        return token_idx

    def _update_node_positions_from_aligned(
        self,
        aligned_expr: str,
        ast: ASTNode,
        node_id_map: Dict[int, ASTNode],
    ) -> None:
        """
        Update the builder's node_positions based on the aligned expression.
        This allows us to track positions correctly for subsequent iterations.
        """
        self.builder.node_positions = {}
        self._find_node_positions_in_string(aligned_expr, ast, 0)

    def _find_node_positions_in_string(
        self,
        expr: str,
        node: ASTNode,
        search_start: int,
    ) -> Tuple[int, int]:
        """
        Find where a node's representation appears in the expression string.
        Returns (start, end) positions.
        """
        node_id = id(node)

        if isinstance(node, NumberNode):
            # Find the number in the string
            num_str = (
                node.original_str if node.original_str else format_number(node.value)
            )
            # Search for the number, accounting for possible padding spaces
            pos = search_start
            while pos < len(expr):
                if expr[pos : pos + len(num_str)] == num_str:
                    self.builder.node_positions[node_id] = (pos, pos + len(num_str))
                    return (pos, pos + len(num_str))
                pos += 1
            # Fallback - just use search_start
            self.builder.node_positions[node_id] = (
                search_start,
                search_start + len(num_str),
            )
            return (search_start, search_start + len(num_str))

        if isinstance(node, BinaryOpNode):
            # Find left child first
            left_start, left_end = self._find_node_positions_in_string(
                expr, node.left, search_start
            )
            # Find operator
            op_pos = expr.find(f" {node.op} ", left_end)
            if op_pos < 0:
                op_pos = left_end
            # Find right child
            right_start, right_end = self._find_node_positions_in_string(
                expr, node.right, op_pos + 3
            )
            # This node spans from left to right
            self.builder.node_positions[node_id] = (left_start, right_end)
            return (left_start, right_end)

        if isinstance(node, UnaryOpNode):
            # Find the operator and operand
            child_start, child_end = self._find_node_positions_in_string(
                expr, node.operand, search_start + 1
            )
            self.builder.node_positions[node_id] = (search_start, child_end)
            return (search_start, child_end)

        if isinstance(node, FunctionNode):
            # Find function name
            func_pos = expr.find(node.name, search_start)
            if func_pos < 0:
                func_pos = search_start
            # Find closing paren
            paren_count = 0
            end_pos = func_pos + len(node.name)
            while end_pos < len(expr):
                if expr[end_pos] == "(":
                    paren_count += 1
                elif expr[end_pos] == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        end_pos += 1
                        break
                end_pos += 1
            self.builder.node_positions[node_id] = (func_pos, end_pos)
            return (func_pos, end_pos)

        return (search_start, search_start)


def visualize_expression(expression: str, compact: bool = True) -> str:
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

    Returns:
        A multi-line string showing the step-by-step evaluation with
        connecting lines indicating the flow.

    Example:
        >>> print(visualize_expression("3 + 5 * (2 - 8) / 4 ^ 2"))
        3 + 5 * (2 - 8) / 4 ^ 2
        │   │   └─────┘   └───┘
        │   │      │   ┌────┘
        3 + 5 *   -6 / 16
        │   └──────┘   │
        │       │   ┌──┘
        3 +   -30 / 16
        │     └──────┘
        │         │
        3 +   -1.875
        └──────────┘
             │
          1.125
    """
    visualizer = ExpressionVisualizer(expression, compact=compact)
    return visualizer.visualize()


def show_evaluation(expression: str, compact: bool = False) -> None:
    """
    Print the step-by-step evaluation of a mathematical expression.

    Convenience function that prints the visualization directly.

    Args:
        expression: A mathematical expression string.
        compact: If True, removes unnecessary whitespace from the output.
    """
    print(visualize_expression(expression, compact=compact))


def evaluate(expression: str) -> float:
    """
    Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression string.

    Returns:
        The numerical result of the expression.
    """
    tokenizer = Tokenizer(expression)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    evaluator = Evaluator()
    return evaluator.evaluate(ast)


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    # Test examples
    test_expressions = [
        "3 + 5 * (2 - 8) / 4 ^ 2",
        "2 + 3 * 4",
        "10 / 2 + 3",
        "2 ^ 3 ^ 2",  # Right-associative: 2^(3^2) = 2^9 = 512
        "sqrt(16) + 2",
        "sin(0) + cos(0)",
        "(1 + 2) * (3 + 4)",
        "2 * 3 + 4 * 5",
        "-0.24384616888471886*0                   + 0.24384616888471886 - 3 + 2.7561538311152813*(1 - 0                  )",
        # "-0.24384616888471886*0                   + 0.24384616888471886 - 3 + 2.7561538311152813*(1 - 0                  ) + (-0.8337335562402163 - 1.3709710108389572*0                   + 3.03843812331939 - 3) + 0.7952954329208262*(1 - 0                  ) + (-1.1928471980738178 - 1.8123132157483224*0                   + 5.354324644092099 - 3) - 1.1614774460182815*(1 - 0                  ) + (-0.3253482844630507 - 0.4503120064825429*0                   + 3.9923234348263192 - 3) - 0.6669751503632684*(1 - 0                  ) + 3*1      + (-0.10250397212815927*0                   + 0.10250397212815927 - 3) + 2.8974960278718407*(1 - 0                  ) + (-0.4829059186112905 - 0.41621591761633714*0.3183098861837907  + 1.3820277548389182 - 3) + 2.1008781637723724*(1 - 1                  ) + (-1.4656922112682929 - 1.4184409594266887*0                   + 4.349825381963274 - 3) + 0.11586682930501846*(1 - 0                  ) + (-1.4912595097946204 - 1.4969761717794265*0                   + 5.8755410303213775 - 3) - 1.3842815205267573*(1 - 0                  ) + (-0.7866767864238279 - 0.4500158226549213*0                   + 4.828580681196872 - 3) - 1.0419038947730437*(1 - 0                  ) - 0.6346872015120248",
    ]

    print("=" * 60)
    print("Expression Evaluation Visualizer")
    print("=" * 60)

    for expr in test_expressions:
        print(f"\nExpression: {expr}")
        print("-" * 40)
        try:
            result = visualize_expression(expr)
            print(result)
            print(f"\nFinal value: {evaluate(expr)}")
            with open("debug_output.txt", "w", encoding="utf-8") as f:
                f.write(f"Expression: {expr}\n")
                f.write(result)
                f.write(f"\nFinal value: {evaluate(expr)}\n")
                f.write("=" * 60 + "\n")
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
        print()
