from .Tokenization import Token, TokenType
from .AST import (
    ASTNode,
    AST,
    NumberNode,
    FunctionNode,
    RelationalExprNode,
    ParenthesesNode,
    AddNode,
    SubtractNode,
    MultiplyNode,
    DivideNode,
    PowerNode,
)
from .EmptyExpressionError import EmptyExpressionError
from typing import List, Optional


class Parser:
    """Parse tokens into an AST using recursive descent with operator precedence."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> AST:
        """Parse tokens into AST."""
        if not self.tokens:
            raise EmptyExpressionError()
        root = self._parse_expression()
        if self.pos < len(self.tokens):
            raise ValueError(
                f"Unexpected token at position {self.tokens[self.pos].start}: {self.tokens[self.pos].value}"
            )
        ast = AST(root)
        # Positions are set during parsing based on original string spans.
        root.assign_parent_relationships()
        return ast

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
        """Parse a full expression, including relational operators (lowest precedence)."""
        left = self._parse_additive()

        token = self._current()
        if token and token.type == TokenType.RELATIONAL:
            op_token = self._consume()
            right = self._parse_additive()
            node = RelationalExprNode(op=op_token.value, left=left, right=right)
            node.startPos = left.startPos
            node.endPos = right.endPos
            return node

        return left

    def _parse_additive(self) -> ASTNode:
        """Parse addition and subtraction (left-associative)."""
        operands = [self._parse_multiplicative()]
        operators = []

        while True:
            token = self._current()
            if token and token.type == TokenType.OPERATOR and token.value in ["+", "-"]:
                op_token = self._consume()
                operators.append(op_token.value)
                operands.append(self._parse_multiplicative())
            else:
                break

        if not operators:
            return operands[0]

        # Group by operator type
        return self._group_operators(operands, operators)

    def _parse_multiplicative(self) -> ASTNode:
        """Parse multiplication and division (left-associative)."""
        operands = [self._parse_power()]
        operators = []

        while True:
            token = self._current()
            if token and token.type == TokenType.OPERATOR and token.value in ["*", "/"]:
                op_token = self._consume()
                operators.append(op_token.value)
                operands.append(self._parse_power())
            else:
                break

        if not operators:
            return operands[0]

        return self._group_operators(operands, operators)

    def _parse_power(self) -> ASTNode:
        """Parse exponentiation (right-associative)."""
        operands = [self._parse_unary()]
        operators = []

        while True:
            token = self._current()
            if (
                token
                and token.type == TokenType.OPERATOR
                and token.value in ["^", "**"]
            ):
                op_token = self._consume()
                operators.append(op_token.value)
                operands.append(self._parse_unary())
            else:
                break

        if not operators:
            return operands[0]

        # Power is right-associative; our node holds all operands in order
        node = PowerNode(operands=operands)
        node.startPos = operands[0].startPos
        node.endPos = operands[-1].endPos
        return node

    def _parse_unary(self) -> ASTNode:
        """Parse unary operators."""
        token = self._current()

        if token and token.type == TokenType.OPERATOR:
            if token.value == "-":
                op_token = self._consume()
                operand = self._parse_unary()
                # Simplify -NumberNode to NumberNode(-value)
                if isinstance(operand, NumberNode):
                    node = NumberNode(value=-operand.value)
                    node.startPos = op_token.start
                    node.endPos = operand.endPos
                    return node
                node = SubtractNode(operands=[operand])
                node.startPos = op_token.start
                node.endPos = operand.endPos
                return node
            elif token.value == "+":
                self._consume()
                return self._parse_unary()

        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        """Parse primary expressions: numbers, functions, and parentheses."""
        token = self._current()

        if token is None:
            raise ValueError("Unexpected end of expression")

        # Number
        if token.type == TokenType.NUMBER:
            num_token = self._consume()
            node = NumberNode(value=float(num_token.value))
            node.startPos = num_token.start
            node.endPos = num_token.end
            return node

        # Function
        if token.type == TokenType.FUNCTION:
            func_token = self._consume()
            lp = self._consume(TokenType.LPAREN)

            args = []
            if self._current() and self._current().type != TokenType.RPAREN:
                args.append(self._parse_additive())
                while self._current() and self._current().type == TokenType.COMMA:
                    self._consume(TokenType.COMMA)
                    args.append(self._parse_additive())

            rp = self._consume(TokenType.RPAREN)
            node = FunctionNode(name=func_token.value, args=args)
            node.startPos = func_token.start
            node.endPos = rp.end
            return node

        # Parentheses
        if token.type == TokenType.LPAREN:
            lp = self._consume(TokenType.LPAREN)
            expr = self._parse_additive()
            rp = self._consume(TokenType.RPAREN)
            node = ParenthesesNode(child=expr)
            node.startPos = lp.start
            node.endPos = rp.end
            return node

        raise ValueError(f"Unexpected token: {token.value} at position {token.start}")

    def _group_operators(
        self, operands: List[ASTNode], operators: List[str]
    ) -> ASTNode:
        """
        Group operands by operator type for left-associative operators.
        This handles expressions like "1 + 2 - 3 + 4" by grouping into appropriate nodes.
        """
        if not operators:
            return operands[0]

        # Build expression left to right
        result = operands[0]
        current_op = operators[0]
        current_operands = [result]

        for i, op in enumerate(operators):
            if op == current_op:
                current_operands.append(operands[i + 1])
            else:
                # Finalize current operator group
                result = self._create_operator_node(current_op, current_operands)
                current_op = op
                current_operands = [result, operands[i + 1]]

        # Finalize last operator group
        result = self._create_operator_node(current_op, current_operands)
        return result

    def _create_operator_node(self, op: str, operands: List[ASTNode]) -> ASTNode:
        """Create the appropriate operator node based on the operator string."""
        if op == "+":
            node = AddNode(operands=operands)
        elif op == "-":
            node = SubtractNode(operands=operands)
        elif op == "*":
            node = MultiplyNode(operands=operands)
        elif op == "/":
            node = DivideNode(operands=operands)
        elif op in ["^", "**"]:
            node = PowerNode(operands=operands)
        else:
            raise ValueError(f"Unknown operator: {op}")
        # Span across first to last operand to cover operator(s) in between
        if operands:
            node.startPos = operands[0].startPos
            node.endPos = operands[-1].endPos
        return node
