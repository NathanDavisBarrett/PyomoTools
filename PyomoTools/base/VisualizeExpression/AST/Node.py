from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import math


class ASTNode(ABC):
    """Base class for AST nodes with position tracking."""

    def __init__(
        self,
        children: Optional[List["ASTNode"]] = None,
        parent_info: Optional[Tuple["ASTNode", int]] = None,
    ):
        """
        Docstring for __init__

        :param children: List of child ASTNodes
        :param parent_info: Tuple of (parent ASTNode, child index in parent)
        """

        self.children = children if children is not None else []
        self.parent_info = parent_info

        # These are only set by the Tree class
        self.startPos = None
        self.endPos = None

    @property
    @abstractmethod
    def display_str(self) -> str:
        """Generate the display string for the node."""
        pass

    @abstractmethod
    def determine_positions(self, start_pos: int = 0) -> int:
        """
        Recursively determine the start and end positions of this node
        and its children in the display string.

        Args:
            start_pos: The starting position for this node.
        Returns:
            The end position after this node's display string.
        """
        pass

    @property
    @abstractmethod
    def evaluation_ready(self) -> bool:
        """Indicates if the node is ready for evaluation (i.e., all children are numbers)."""
        pass

    @abstractmethod
    def evaluate(self) -> "NumberNode":
        """Evaluate the node and return a NumberNode with the result."""
        assert self.evaluation_ready, "Node is not ready for evaluation."

    def DFS(self, stop_at_eval_ready: bool = False):
        if not stop_at_eval_ready or not self.evaluation_ready:
            for child in self.children:
                yield from child.DFS(stop_at_eval_ready)

        yield self

    def LeafNodes(self):
        if not self.children:
            yield self
        for child in self.children:
            yield from child.LeafNodes()

    def assign_parent_relationships(self):
        for i, child in enumerate(self.children):
            child.parent_info = (self, i)
            child.assign_parent_relationships()


class NumberNode(ASTNode):
    def __init__(self, value: float, parent_info: Optional[ASTNode] = None):
        super().__init__(children=[], parent_info=parent_info)
        self.value = value
        self.display_format = "{}"  # Default format for numbers

    @property
    def display_str(self) -> str:
        """Generate the display string for the number."""
        return self.display_format.format(self.value)

    def determine_positions(self, start_pos: int = 0) -> int:
        """
        Recursively determine the start and end positions of this node
        and its children in the display string.

        Args:
            start_pos: The starting position for this node.
        Returns:
            The end position after this node's display string.
        """
        self.startPos = start_pos
        self.endPos = start_pos + len(self.display_str)
        return self.endPos

    @property
    def evaluation_ready(self) -> bool:
        """Number nodes are always ready for evaluation."""
        return True

    def evaluate(self):
        super().evaluate()
        return self


class Associativity(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Operator:
    str: str
    precedence: int
    associativity: Associativity


ADD = Operator("+", 1, Associativity.LEFT)
SUBTRACT = Operator("-", 2, Associativity.LEFT)
MULTIPLY = Operator("*", 3, Associativity.LEFT)
DIVIDE = Operator("/", 4, Associativity.LEFT)
POWER = Operator("^", 5, Associativity.RIGHT)


class OperatorNode(ASTNode):
    """
    A node for repeated operators.

    For example, in the expression "1 + 2 + 3 + 4", this node would represent
    the 'ADD' operator with operands [1, 2, 3, 4].

    Or the expression "2 ^ 3 ^ 4" would be represented as a 'POWER' operator node
    with operands [2, 3, 4].

    Order of operations is preserved by the order of operands in the list.

    If the operator has only one operand, it is treated as a unary operator.
    """

    def __init__(
        self,
        operator: Operator,
        operands: List[ASTNode],
        parent_info: Optional[ASTNode] = None,
    ):
        super().__init__(children=operands, parent_info=parent_info)
        self.operator = operator

    @property
    def operands(self) -> List[ASTNode]:
        return self.children

    @property
    def display_str(self) -> str:
        """Generate the display string for the operator node."""
        if len(self.operands) == 1:
            # Unary operator
            return f"{self.operator.str}{self.operands[0].display_str}"
        else:
            return f" {self.operator.str} ".join(
                [operand.display_str for operand in self.operands]
            )

    def determine_positions(self, start_pos: int = 0) -> int:
        """
        Recursively determine the start and end positions of this node
        and its children in the display string.

        Args:
            start_pos: The starting position for this node.
        Returns:
            The end position after this node's display string.
        """
        self.startPos = start_pos
        current_pos = start_pos

        if len(self.operands) == 1:
            # Unary operator
            current_pos += len(self.operator.str)  # Operator

        for i, operand in enumerate(self.operands):
            current_pos = operand.determine_positions(current_pos)
            if i < len(self.operands) - 1:
                # Add space for operator and surrounding spaces
                current_pos += len(f" {self.operator.str} ")

        self.endPos = current_pos
        return self.endPos

    @property
    def evaluation_ready(self) -> bool:
        """Indicates if all operands are ready for evaluation."""
        return all(isinstance(operand, NumberNode) for operand in self.operands)


class AddNode(OperatorNode):
    def __init__(self, operands, parent_info: Optional[ASTNode] = None):
        super().__init__(ADD, operands, parent_info=parent_info)

    def evaluate(self):
        super().evaluate()
        total = sum(operand.value for operand in self.operands)
        return NumberNode(value=total, parent_info=self.parent_info)


class SubtractNode(OperatorNode):
    def __init__(self, operands, parent_info: Optional[ASTNode] = None):
        super().__init__(SUBTRACT, operands, parent_info=parent_info)

    def evaluate(self):
        super().evaluate()
        if len(self.operands) == 1:
            # Unary minus
            result = -self.operands[0].value
        else:
            result = self.operands[0].value
            for operand in self.operands[1:]:
                result -= operand.value
        return NumberNode(value=result, parent_info=self.parent_info)


class MultiplyNode(OperatorNode):
    def __init__(self, operands, parent_info: Optional[ASTNode] = None):
        super().__init__(MULTIPLY, operands, parent_info=parent_info)

    def evaluate(self):
        super().evaluate()
        product = 1.0
        for operand in self.operands:
            product *= operand.value
        return NumberNode(value=product, parent_info=self.parent_info)


class DivideNode(OperatorNode):
    def __init__(self, operands, parent_info: Optional[ASTNode] = None):
        super().__init__(DIVIDE, operands, parent_info=parent_info)

    def evaluate(self):
        super().evaluate()
        if len(self.operands) == 1:
            # Unary reciprocal
            result = 1.0 / self.operands[0].value
        else:
            result = self.operands[0].value
            for operand in self.operands[1:]:
                result /= operand.value
        return NumberNode(value=result, parent_info=self.parent_info)


class PowerNode(OperatorNode):
    def __init__(self, operands, parent_info: Optional[ASTNode] = None):
        super().__init__(POWER, operands, parent_info=parent_info)

    def evaluate(self):
        super().evaluate()
        result = self.operands[0].value
        for operand in self.operands[1:]:
            result **= operand.value
        return NumberNode(value=result, parent_info=self.parent_info)


class FunctionNode(ASTNode):
    def __init__(self, name, args=None, parent_info: Optional[ASTNode] = None):
        super().__init__(args if args is not None else [], parent_info=parent_info)
        self.name = name

    @property
    def args(self) -> List[ASTNode]:
        return self.children

    @property
    def display_str(self) -> str:
        """Generate the display string for the function node."""
        arg_strs = [arg.display_str for arg in self.args]
        return f"{self.name}({', '.join(arg_strs)})"

    def determine_positions(self, start_pos: int = 0) -> int:
        """
        Recursively determine the start and end positions of this node
        and its children in the display string.

        Args:
            start_pos: The starting position for this node.
        Returns:
            The end position after this node's display string.
        """
        self.startPos = start_pos
        current_pos = start_pos

        # Function name and opening parent_infohesis
        current_pos += len(self.name) + 1  # +1 for '('

        for i, arg in enumerate(self.args):
            current_pos = arg.determine_positions(current_pos)
            if i < len(self.args) - 1:
                current_pos += 2  # ', '

        # Closing parent_infohesis
        current_pos += 1

        self.endPos = current_pos
        return self.endPos

    @property
    def evaluation_ready(self) -> bool:
        """Indicates if all arguments are ready for evaluation."""
        return all(isinstance(arg, NumberNode) for arg in self.args)

    def evaluate(self):
        super().evaluate()

        evaluated_args = [arg.value for arg in self.args]

        if not hasattr(math, self.name):
            raise ValueError(f"Unknown function: {self.name}. Cannot evaluate.")

        func = getattr(math, self.name)
        result = func(*evaluated_args)

        return NumberNode(value=result, parent_info=self.parent_info)


class ParenthesesNode(ASTNode):
    """
    A node representing parent_infoheses in the expression.

    This node wraps a single child ASTNode that represents the expression
    inside the parent_infoheses.
    """

    def __init__(self, child: ASTNode, parent_info: Optional[ASTNode] = None):
        super().__init__(children=[child], parent_info=parent_info)

    @property
    def child(self) -> ASTNode:
        return self.children[0]

    @property
    def display_str(self) -> str:
        """Generate the display string for the parent_infoheses node."""
        return f"({self.child.display_str})"

    def determine_positions(self, start_pos: int = 0) -> int:
        """
        Recursively determine the start and end positions of this node
        and its children in the display string.

        Args:
            start_pos: The starting position for this node.
        Returns:
            The end position after this node's display string.
        """
        self.startPos = start_pos
        current_pos = start_pos

        # Opening parent_infohesis
        current_pos += 1

        # Child node
        current_pos = self.child.determine_positions(current_pos)

        # Closing parent_infohesis
        current_pos += 1

        self.endPos = current_pos
        return self.endPos

    @property
    def evaluation_ready(self) -> bool:
        """Indicates if the child node is ready for evaluation."""
        return isinstance(self.child, NumberNode)

    def evaluate(self):
        super().evaluate()
        return NumberNode(value=self.child.value, parent_info=self.parent_info)


class RelationalExprNode(ASTNode):
    """
    AST node for relational expressions (e.g., '3 + 5 == 2 * 4').

    The left and right sides are treated as independent ASTs for evaluation
    purposes - no expressions are transferred across the relational operator.
    However, for display/visualization purposes, they are rendered together.
    """

    def __init__(
        self,
        op: str,
        left: ASTNode,
        right: ASTNode,
        parent_info: Optional[ASTNode] = None,
    ):
        super().__init__(children=[left, right], parent_info=parent_info)
        self.op = op  # The relational operator: ==, <=, >=, <, >, !=

    @property
    def left(self) -> ASTNode:
        return self.children[0]

    @property
    def right(self) -> ASTNode:
        return self.children[1]

    @property
    def display_str(self) -> str:
        """Generate the display string for the relational expression."""
        return f"{self.left.display_str} {self.op} {self.right.display_str}"

    def determine_positions(self, start_pos: int = 0) -> int:
        """
        Recursively determine the start and end positions of this node
        and its children in the display string.

        Args:
            start_pos: The starting position for this node.
        Returns:
            The end position after this node's display string.
        """
        self.startPos = start_pos
        current_pos = start_pos

        # Left side
        current_pos = self.left.determine_positions(current_pos)

        # Operator and surrounding spaces
        current_pos += len(f" {self.op} ")

        # Right side
        current_pos = self.right.determine_positions(current_pos)

        self.endPos = current_pos
        return self.endPos

    @property
    def evaluation_ready(self) -> bool:
        """Indicates if both sides are ready for evaluation."""
        return isinstance(self.left, NumberNode) and isinstance(self.right, NumberNode)

    def evaluate(self, tolerance: float = 1e-9) -> NumberNode:
        super().evaluate()

        left_value = self.left.value
        right_value = self.right.value

        if self.op == "==":
            result = abs(left_value - right_value) <= tolerance
        elif self.op == "!=":
            result = abs(left_value - right_value) > tolerance
        elif self.op == "<":
            result = left_value < right_value + tolerance
        elif self.op == "<=":
            result = left_value <= right_value + tolerance
        elif self.op == ">":
            result = left_value > right_value - tolerance
        elif self.op == ">=":
            result = left_value >= right_value - tolerance
        else:
            raise ValueError(f"Unknown relational operator: {self.op}")

        return NumberNode(value=result, parent_info=self.parent_info)
