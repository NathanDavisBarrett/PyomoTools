import pytest
from ..Tokenization import Tokenizer
from ..Parser import Parser
from ..AST import (
    NumberNode,
    AddNode,
    SubtractNode,
    MultiplyNode,
    DivideNode,
    PowerNode,
    FunctionNode,
    ParenthesesNode,
    RelationalExprNode,
)
import math


class TestParser:
    """Test suite for the Parser class."""

    def _parse(self, expression: str):
        """Helper method to tokenize and parse an expression."""
        tokenizer = Tokenizer(expression)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def test_simple_number(self):
        """Test parsing a simple number."""
        ast = self._parse("42")
        assert isinstance(ast.root, NumberNode)
        assert ast.root.value == 42.0

    def test_decimal_number(self):
        """Test parsing a decimal number."""
        ast = self._parse("3.14")
        assert isinstance(ast.root, NumberNode)
        assert ast.root.value == pytest.approx(3.14)

    def test_simple_addition(self):
        """Test parsing simple addition."""
        ast = self._parse("2 + 3")
        assert isinstance(ast.root, AddNode)
        assert len(ast.root.operands) == 2
        assert ast.root.operands[0].value == 2.0
        assert ast.root.operands[1].value == 3.0

    def test_simple_subtraction(self):
        """Test parsing simple subtraction."""
        ast = self._parse("5 - 3")
        assert isinstance(ast.root, SubtractNode)
        assert len(ast.root.operands) == 2
        assert ast.root.operands[0].value == 5.0
        assert ast.root.operands[1].value == 3.0

    def test_simple_multiplication(self):
        """Test parsing simple multiplication."""
        ast = self._parse("4 * 5")
        assert isinstance(ast.root, MultiplyNode)
        assert len(ast.root.operands) == 2
        assert ast.root.operands[0].value == 4.0
        assert ast.root.operands[1].value == 5.0

    def test_simple_division(self):
        """Test parsing simple division."""
        ast = self._parse("10 / 2")
        assert isinstance(ast.root, DivideNode)
        assert len(ast.root.operands) == 2
        assert ast.root.operands[0].value == 10.0
        assert ast.root.operands[1].value == 2.0

    def test_simple_power(self):
        """Test parsing simple exponentiation."""
        ast = self._parse("2 ^ 3")
        assert isinstance(ast.root, PowerNode)
        assert len(ast.root.operands) == 2
        assert ast.root.operands[0].value == 2.0
        assert ast.root.operands[1].value == 3.0

    def test_chained_addition(self):
        """Test parsing chained additions."""
        ast = self._parse("1 + 2 + 3 + 4")
        assert isinstance(ast.root, AddNode)
        assert len(ast.root.operands) == 4
        assert [op.value for op in ast.root.operands] == [1.0, 2.0, 3.0, 4.0]

    def test_chained_subtraction(self):
        """Test parsing chained subtractions."""
        ast = self._parse("10 - 2 - 3")
        assert isinstance(ast.root, SubtractNode)
        assert len(ast.root.operands) == 3
        assert [op.value for op in ast.root.operands] == [10.0, 2.0, 3.0]

    def test_mixed_addition_subtraction(self):
        """Test parsing mixed addition and subtraction."""
        ast = self._parse("1 + 2 - 3 + 4")
        # Should create separate nodes for each operator type
        assert isinstance(ast.root, AddNode)

    def test_operator_precedence_mult_add(self):
        """Test that multiplication has higher precedence than addition."""
        ast = self._parse("2 + 3 * 4")
        assert isinstance(ast.root, AddNode)
        assert len(ast.root.operands) == 2
        assert isinstance(ast.root.operands[0], NumberNode)
        assert isinstance(ast.root.operands[1], MultiplyNode)

    def test_operator_precedence_div_sub(self):
        """Test that division has higher precedence than subtraction."""
        ast = self._parse("10 - 6 / 2")
        assert isinstance(ast.root, SubtractNode)
        assert len(ast.root.operands) == 2
        assert isinstance(ast.root.operands[0], NumberNode)
        assert isinstance(ast.root.operands[1], DivideNode)

    def test_operator_precedence_power_mult(self):
        """Test that exponentiation has higher precedence than multiplication."""
        ast = self._parse("2 * 3 ^ 4")
        assert isinstance(ast.root, MultiplyNode)
        assert len(ast.root.operands) == 2
        assert isinstance(ast.root.operands[0], NumberNode)
        assert isinstance(ast.root.operands[1], PowerNode)

    def test_parentheses_override_precedence(self):
        """Test that parentheses override operator precedence."""
        ast = self._parse("(2 + 3) * 4")
        assert isinstance(ast.root, MultiplyNode)
        assert len(ast.root.operands) == 2
        assert isinstance(ast.root.operands[0], ParenthesesNode)
        assert isinstance(ast.root.operands[1], NumberNode)

    def test_nested_parentheses(self):
        """Test parsing nested parentheses."""
        ast = self._parse("((2 + 3))")
        assert isinstance(ast.root, ParenthesesNode)
        assert isinstance(ast.root.child, ParenthesesNode)
        assert isinstance(ast.root.child.child, AddNode)

    def test_unary_minus(self):
        """Test parsing unary minus."""
        ast = self._parse("-5")
        assert isinstance(ast.root, SubtractNode)
        assert len(ast.root.operands) == 1
        assert ast.root.operands[0].value == 5.0

    def test_unary_plus(self):
        """Test parsing unary plus (should be ignored)."""
        ast = self._parse("+5")
        assert isinstance(ast.root, NumberNode)
        assert ast.root.value == 5.0

    def test_double_negative(self):
        """Test parsing double negative."""
        ast = self._parse("-(-5)")
        assert isinstance(ast.root, SubtractNode)
        assert len(ast.root.operands) == 1
        assert isinstance(ast.root.operands[0], ParenthesesNode)

    def test_unary_in_expression(self):
        """Test parsing unary operator in an expression."""
        ast = self._parse("3 + -2")
        assert isinstance(ast.root, AddNode)
        assert len(ast.root.operands) == 2
        assert isinstance(ast.root.operands[1], SubtractNode)

    def test_simple_function(self):
        """Test parsing a simple function call."""
        ast = self._parse("sin(0)")
        assert isinstance(ast.root, FunctionNode)
        assert ast.root.name == "sin"
        assert len(ast.root.args) == 1
        assert ast.root.args[0].value == 0.0

    def test_function_with_expression(self):
        """Test parsing a function with an expression as argument."""
        ast = self._parse("sin(2 + 3)")
        assert isinstance(ast.root, FunctionNode)
        assert ast.root.name == "sin"
        assert len(ast.root.args) == 1
        assert isinstance(ast.root.args[0], AddNode)

    def test_function_with_multiple_args(self):
        """Test parsing a function with multiple arguments."""
        ast = self._parse("atan(1, 2)")
        assert isinstance(ast.root, FunctionNode)
        assert ast.root.name == "atan"
        assert len(ast.root.args) == 2
        assert ast.root.args[0].value == 1.0
        assert ast.root.args[1].value == 2.0

    def test_nested_functions(self):
        """Test parsing nested function calls."""
        ast = self._parse("sin(cos(0))")
        assert isinstance(ast.root, FunctionNode)
        assert ast.root.name == "sin"
        assert isinstance(ast.root.args[0], FunctionNode)
        assert ast.root.args[0].name == "cos"

    def test_function_in_expression(self):
        """Test parsing a function as part of an expression."""
        ast = self._parse("2 * sin(0) + 1")
        assert isinstance(ast.root, AddNode)
        assert isinstance(ast.root.operands[0], MultiplyNode)
        assert isinstance(ast.root.operands[0].operands[1], FunctionNode)

    def test_relational_equality(self):
        """Test parsing equality relational expression."""
        ast = self._parse("3 + 5 == 2 * 4")
        assert isinstance(ast.root, RelationalExprNode)
        assert ast.root.op == "=="
        assert isinstance(ast.root.left, AddNode)
        assert isinstance(ast.root.right, MultiplyNode)

    def test_relational_less_than(self):
        """Test parsing less-than relational expression."""
        ast = self._parse("2 < 5")
        assert isinstance(ast.root, RelationalExprNode)
        assert ast.root.op == "<"
        assert ast.root.left.value == 2.0
        assert ast.root.right.value == 5.0

    def test_relational_greater_equal(self):
        """Test parsing greater-than-or-equal relational expression."""
        ast = self._parse("10 >= 5")
        assert isinstance(ast.root, RelationalExprNode)
        assert ast.root.op == ">="
        assert ast.root.left.value == 10.0
        assert ast.root.right.value == 5.0

    def test_relational_is_root_for_complex_sides(self):
        """Relational expression must be the root even with complex sides."""
        expr = "2 + 3 * (4 - 5) ^ 2 <= sqrt(16) + 7 / (1 + 1)"
        ast = self._parse(expr)
        assert isinstance(ast.root, RelationalExprNode)

        def has_relational(node):
            if isinstance(node, RelationalExprNode):
                return True
            for child in getattr(node, "children", []) or []:
                if has_relational(child):
                    return True
            return False

        assert not has_relational(ast.root.left)
        assert not has_relational(ast.root.right)

    def test_relational_is_root_with_functions_and_powers(self):
        """Relational root holds for function and power-heavy expressions."""
        expr = "sin(2 + 3 ^ 2) > cos(4) - 5 ^ (1 + 1)"
        ast = self._parse(expr)
        assert isinstance(ast.root, RelationalExprNode)

        def has_relational(node):
            if isinstance(node, RelationalExprNode):
                return True
            for child in getattr(node, "children", []) or []:
                if has_relational(child):
                    return True
            return False

        assert not has_relational(ast.root.left)
        assert not has_relational(ast.root.right)

    def test_complex_expression(self):
        """Test parsing a complex mathematical expression."""
        ast = self._parse("2 * sin(3.14 / 2) + sqrt(16) ^ 0.5")
        assert isinstance(ast.root, AddNode)
        # Just verify it parses without error and has correct structure
        assert len(ast.root.operands) == 2

    def test_expression_with_constants(self):
        """Test parsing expressions with mathematical constants."""
        ast = self._parse("2 * pi + e")
        assert isinstance(ast.root, AddNode)
        assert isinstance(ast.root.operands[0], MultiplyNode)

    def test_right_associative_power(self):
        """Test that power operator is right-associative."""
        ast = self._parse("2 ^ 3 ^ 2")
        assert isinstance(ast.root, PowerNode)
        # For right associativity: 2 ^ (3 ^ 2) = 2 ^ 9 = 512
        # The operands should be [2, 3, 2] and evaluated right-to-left
        assert len(ast.root.operands) == 3

    def test_empty_expression_error(self):
        """Test that empty expression raises an error."""
        with pytest.raises(ValueError, match="Empty expression"):
            self._parse("")

    def test_unexpected_token_error(self):
        """Test that unexpected token raises an error."""
        tokenizer = Tokenizer("2 + 3 4")
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        with pytest.raises(ValueError, match="Unexpected token"):
            parser.parse()

    def test_missing_closing_paren_error(self):
        """Test that missing closing parenthesis raises an error."""
        with pytest.raises(ValueError):
            self._parse("(2 + 3")

    def test_missing_function_args_error(self):
        """Test that function without parentheses raises an error."""
        with pytest.raises(ValueError):
            self._parse("sin")

    def test_display_str_simple(self):
        """Test display string generation for simple expression."""
        ast = self._parse("2 + 3")
        display = ast.display_str
        assert "2" in display
        assert "+" in display
        assert "3" in display

    def test_display_str_complex(self):
        """Test display string generation for complex expression."""
        ast = self._parse("2 * (3 + 4)")
        display = ast.display_str
        assert "2" in display
        assert "*" in display
        assert "(" in display
        assert "+" in display
        assert ")" in display

    def test_position_assignment(self):
        """Test that positions are assigned to nodes."""
        ast = self._parse("2 + 3")
        assert ast.root.startPos is not None
        assert ast.root.endPos is not None

    def test_parent_relationships(self):
        """Test that parent relationships are assigned."""
        ast = self._parse("2 + 3")
        # Root should have no parent
        assert ast.root.parent_info is None
        # Children should have parent info
        for child in ast.root.children:
            assert child.parent_info is not None
            assert child.parent_info[0] == ast.root

    # --- Position tests ---

    def test_positions_number_simple(self):
        ast = self._parse("42")
        assert ast.root.startPos == 0
        assert ast.root.endPos == 2

    def test_positions_add_no_spaces(self):
        expr = "12+3"
        ast = self._parse(expr)
        # 12(0-2) + (2-3) 3(3-4) => add spans 0-4
        assert ast.root.startPos == 0
        assert ast.root.endPos == 4

    def test_positions_add_with_spaces(self):
        expr = "  1 + 23 "
        ast = self._parse(expr)
        # '1' at 2-3, '23' at 6-8 => add spans 2-8
        assert ast.root.startPos == 2
        assert ast.root.endPos == 8

    def test_positions_parentheses(self):
        expr = "(2+3)"
        ast = self._parse(expr)
        # Root is parentheses
        assert isinstance(ast.root, ParenthesesNode)
        assert ast.root.startPos == 0
        assert ast.root.endPos == 5
        # Inner add spans 1-4
        inner = ast.root.child
        assert inner.startPos == 1
        assert inner.endPos == 4

    def test_positions_function_single_arg(self):
        expr = "sin(10)"
        ast = self._parse(expr)
        fn = ast.root
        assert isinstance(fn, FunctionNode)
        # sin(10): sin 0-3, '(' 3-4, '10' 4-6, ')' 6-7 => 0-7
        assert fn.startPos == 0
        assert fn.endPos == 7
        assert fn.args[0].startPos == 4
        assert fn.args[0].endPos == 6

    def test_positions_unary_minus(self):
        expr = "-5"
        ast = self._parse(expr)
        node = ast.root
        assert node.startPos == 0
        assert node.endPos == 2

    def test_positions_power_chain(self):
        expr = "2^3^4"
        ast = self._parse(expr)
        node = ast.root
        # 2(0-1) ^ 3(2-3) ^ 4(4-5) => power spans 0-5
        assert node.startPos == 0
        assert node.endPos == 5

    def test_positions_relational(self):
        expr = "1+2<=3*4"
        ast = self._parse(expr)
        root = ast.root
        assert isinstance(root, RelationalExprNode)
        # left add: 1(0-1) + 2(2-3) => 0-3
        assert root.left.startPos == 0
        assert root.left.endPos == 3
        # right mult: 3(5-6) * 4(7-8) => 5-8
        assert root.right.startPos == 5
        assert root.right.endPos == 8
        # whole relation spans 0-8
        assert root.startPos == 0
        assert root.endPos == 8

    def test_positions_complex_relational(self):
        expr = "2 + 3 * (4 - 5) ^ 2 <= sqrt(16) + 7 / (1 + 1)"
        ast = self._parse(expr)
        root = ast.root
        assert isinstance(root, RelationalExprNode)
        # start at index of first non-space
        first_idx = next(i for i, ch in enumerate(expr) if not ch.isspace())
        last_idx = max(i for i, ch in enumerate(expr) if not ch.isspace()) + 1
        assert root.startPos == first_idx
        assert root.endPos == last_idx
