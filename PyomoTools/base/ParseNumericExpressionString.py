import math
from typing import List, Tuple, Optional, Dict

print("MAKE RELATIVE!")
from DetermineExpressionConnectors import generate_connectors_and_aligned_expression
from dataclasses import dataclass, field
from enum import Enum, auto

import matplotlib.pyplot as plt
import networkx as nx


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


@dataclass
class ASTNode:
    """Base class for AST nodes with position tracking."""

    start_pos: int = 0
    end_pos: int = 0

    def plot(
        self,
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 2000,
        font_size: int = 10,
        title: str = "AST Tree",
    ):
        """
        Plot the AST tree using matplotlib and networkx.

        Args:
            figsize: Figure size as (width, height) tuple
            node_size: Size of the nodes in the plot
            font_size: Font size for node labels
            title: Title of the plot
        """
        G = nx.DiGraph()
        labels = {}
        node_colors = []

        # Color scheme for different node types
        color_map = {
            "NumberNode": "#90EE90",  # Light green
            "BinaryOpNode": "#FFB6C1",  # Light pink
            "UnaryOpNode": "#DDA0DD",  # Plum
            "FunctionNode": "#87CEEB",  # Sky blue
        }

        def get_node_label(node: "ASTNode") -> str:
            """Get a display label for a node."""
            if isinstance(node, NumberNode):
                return node.original_str if node.original_str else str(node.value)
            elif isinstance(node, BinaryOpNode):
                return node.op
            elif isinstance(node, UnaryOpNode):
                return f"unary {node.op}"
            elif isinstance(node, FunctionNode):
                return f"{node.name}()"
            return "?"

        def add_nodes_edges(
            node: "ASTNode", parent_id: Optional[int] = None, counter: List[int] = None
        ) -> int:
            """Recursively add nodes and edges to the graph."""
            if counter is None:
                counter = [0]

            node_id = counter[0]
            counter[0] += 1

            # Add node
            G.add_node(node_id)
            labels[node_id] = get_node_label(node)
            node_colors.append(color_map.get(type(node).__name__, "#FFFFFF"))

            # Add edge from parent
            if parent_id is not None:
                G.add_edge(parent_id, node_id)

            # Recursively add children
            if isinstance(node, BinaryOpNode):
                add_nodes_edges(node.left, node_id, counter)
                add_nodes_edges(node.right, node_id, counter)
            elif isinstance(node, UnaryOpNode):
                add_nodes_edges(node.operand, node_id, counter)
            elif isinstance(node, FunctionNode):
                for arg in node.args:
                    add_nodes_edges(arg, node_id, counter)

            return node_id

        # Build the graph
        add_nodes_edges(self)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use a hierarchical layout
        pos = _hierarchy_pos(G, 0)

        # Draw the graph
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            font_weight="bold",
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            edge_color="gray",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=15,
                label=name.replace("Node", ""),
            )
            for name, color in color_map.items()
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        plt.tight_layout()
        plt.show()

        return fig, ax


def _hierarchy_pos(
    G: nx.DiGraph,
    root: int,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    xcenter: float = 0.5,
) -> Dict[int, Tuple[float, float]]:
    """
    Position nodes in a tree layout.

    This is a helper function that creates a hierarchical layout for the tree,
    positioning parent nodes above their children.

    Args:
        G: The directed graph (tree)
        root: The root node
        width: Horizontal space allocated for this branch
        vert_gap: Gap between levels
        vert_loc: Vertical location of root
        xcenter: Horizontal location of root

    Returns:
        Dictionary mapping node to (x, y) position
    """
    pos = {root: (xcenter, vert_loc)}
    children = list(G.successors(root))

    if children:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos.update(
                _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                )
            )
    return pos


# Operators that are associative and can be rebalanced
ASSOCIATIVE_OPS = {"+", "*"}


def balance_tree(node: "ASTNode") -> "ASTNode":
    """
    Balance an AST tree to minimize depth while preserving order of operations.

    For associative operators (+ and *), this flattens chains of the same operator
    and rebuilds them as a balanced binary tree. Non-associative operators (-, /, ^)
    are left unchanged to preserve correct semantics.

    Args:
        node: The root of the AST to balance

    Returns:
        A new balanced AST tree
    """
    if isinstance(node, NumberNode):
        # Leaf node - return a copy
        return NumberNode(
            start_pos=node.start_pos,
            end_pos=node.end_pos,
            value=node.value,
            original_str=node.original_str,
        )

    elif isinstance(node, UnaryOpNode):
        # Balance the operand recursively
        return UnaryOpNode(
            start_pos=node.start_pos,
            end_pos=node.end_pos,
            op=node.op,
            operand=balance_tree(node.operand),
        )

    elif isinstance(node, FunctionNode):
        # Balance all arguments recursively
        return FunctionNode(
            start_pos=node.start_pos,
            end_pos=node.end_pos,
            name=node.name,
            args=[balance_tree(arg) for arg in node.args],
        )

    elif isinstance(node, BinaryOpNode):
        if node.op in ASSOCIATIVE_OPS:
            # Flatten the chain of same operators
            operands = _flatten_associative_chain(node, node.op)
            # Balance the operands recursively first
            balanced_operands = [balance_tree(op) for op in operands]
            # Build a balanced tree from the operands
            return _build_balanced_tree(balanced_operands, node.op)
        else:
            # Non-associative operator - just balance children
            return BinaryOpNode(
                start_pos=node.start_pos,
                end_pos=node.end_pos,
                op=node.op,
                left=balance_tree(node.left),
                right=balance_tree(node.right),
            )

    return node


def _flatten_associative_chain(node: "ASTNode", op: str) -> List["ASTNode"]:
    """
    Flatten a chain of associative binary operations into a list of operands.

    For example, ((1 + 2) + 3) + 4 becomes [1, 2, 3, 4]

    Args:
        node: The node to flatten
        op: The operator we're flattening

    Returns:
        List of operand nodes in left-to-right order
    """
    if not isinstance(node, BinaryOpNode) or node.op != op:
        return [node]

    # Recursively flatten left and right
    left_operands = _flatten_associative_chain(node.left, op)
    right_operands = _flatten_associative_chain(node.right, op)

    return left_operands + right_operands


def _build_balanced_tree(operands: List["ASTNode"], op: str) -> "ASTNode":
    """
    Build a balanced binary tree from a list of operands.

    Uses a divide-and-conquer approach to create a tree of minimal depth.

    Args:
        operands: List of operand nodes
        op: The binary operator to use

    Returns:
        Root of a balanced binary tree
    """
    if len(operands) == 1:
        return operands[0]

    if len(operands) == 2:
        return BinaryOpNode(
            start_pos=operands[0].start_pos,
            end_pos=operands[1].end_pos,
            op=op,
            left=operands[0],
            right=operands[1],
        )

    # Split in the middle and recursively build
    mid = len(operands) // 2
    left_tree = _build_balanced_tree(operands[:mid], op)
    right_tree = _build_balanced_tree(operands[mid:], op)

    return BinaryOpNode(
        start_pos=left_tree.start_pos,
        end_pos=right_tree.end_pos,
        op=op,
        left=left_tree,
        right=right_tree,
    )


def tree_depth(node: "ASTNode") -> int:
    """
    Calculate the depth of an AST tree.

    Args:
        node: The root of the tree

    Returns:
        The depth (number of levels) in the tree
    """
    if isinstance(node, NumberNode):
        return 1
    elif isinstance(node, UnaryOpNode):
        return 1 + tree_depth(node.operand)
    elif isinstance(node, FunctionNode):
        if not node.args:
            return 1
        return 1 + max(tree_depth(arg) for arg in node.args)
    elif isinstance(node, BinaryOpNode):
        return 1 + max(tree_depth(node.left), tree_depth(node.right))
    return 1


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


# tokenizer = Tokenizer("1 * -2 * 3 + 4 * 5 * 6 / 7 + sin(4) + 4 + 2 + 8 + 3 + 5")
# tokens = tokenizer.tokenize()
# parser = Parser(tokens)
# ast = parser.parse()

# print(f"Original tree depth: {tree_depth(ast)}")
# ast.plot(title="Original AST (Unbalanced)")

# balanced_ast = balance_tree(ast)
# print(f"Balanced tree depth: {tree_depth(balanced_ast)}")
# balanced_ast.plot(title="Balanced AST")

# plt.show()
