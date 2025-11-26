"""
Unit tests for the ParseNumericExpressionString module.

Tests cover tokenization, parsing, evaluation, and visualization of
mathematical expressions.
"""

import pytest
import math
from PyomoTools.base.ParseNumericExpressionString import (
    Tokenizer,
    TokenType,
    Parser,
    ExpressionVisualizer,
    visualize_expression,
    show_evaluation,
    evaluate,
    format_number,
    NumberNode,
    BinaryOpNode,
    UnaryOpNode,
    FunctionNode,
)


# =============================================================================
# Tokenizer Tests
# =============================================================================


class TestTokenizer:
    """Tests for the Tokenizer class."""

    def test_tokenize_simple_number(self):
        """Test tokenizing a simple integer."""
        tokenizer = Tokenizer("42")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"

    def test_tokenize_decimal_number(self):
        """Test tokenizing a decimal number."""
        tokenizer = Tokenizer("3.14159")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "3.14159"

    def test_tokenize_scientific_notation(self):
        """Test tokenizing numbers in scientific notation."""
        tokenizer = Tokenizer("1.5e10")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "1.5e10"

    def test_tokenize_operators(self):
        """Test tokenizing all operators."""
        tokenizer = Tokenizer("+ - * / ^")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 5
        for token in tokens:
            assert token.type == TokenType.OPERATOR
        assert [t.value for t in tokens] == ["+", "-", "*", "/", "^"]

    def test_tokenize_parentheses(self):
        """Test tokenizing parentheses."""
        tokenizer = Tokenizer("()")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 2
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.RPAREN

    def test_tokenize_function(self):
        """Test tokenizing function names."""
        tokenizer = Tokenizer("sin(0)")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 4
        assert tokens[0].type == TokenType.FUNCTION
        assert tokens[0].value == "sin"

    def test_tokenize_expression(self):
        """Test tokenizing a complete expression."""
        tokenizer = Tokenizer("3 + 5 * 2")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 5
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[1].type == TokenType.OPERATOR
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[3].type == TokenType.OPERATOR
        assert tokens[4].type == TokenType.NUMBER

    def test_tokenize_constants(self):
        """Test tokenizing mathematical constants."""
        tokenizer = Tokenizer("pi + e")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 3
        assert tokens[0].type == TokenType.NUMBER
        assert float(tokens[0].value) == pytest.approx(math.pi)
        assert tokens[2].type == TokenType.NUMBER
        assert float(tokens[2].value) == pytest.approx(math.e)

    def test_tokenize_unknown_identifier_raises(self):
        """Test that unknown identifiers raise an error."""
        tokenizer = Tokenizer("foo")
        with pytest.raises(ValueError, match="Unknown identifier"):
            tokenizer.tokenize()

    def test_tokenize_unexpected_character_raises(self):
        """Test that unexpected characters raise an error."""
        tokenizer = Tokenizer("3 @ 5")
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenizer.tokenize()


# =============================================================================
# Parser Tests
# =============================================================================


class TestParser:
    """Tests for the Parser class."""

    def test_parse_single_number(self):
        """Test parsing a single number."""
        tokenizer = Tokenizer("42")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, NumberNode)
        assert ast.value == 42.0

    def test_parse_binary_addition(self):
        """Test parsing addition."""
        tokenizer = Tokenizer("3 + 5")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"
        assert isinstance(ast.left, NumberNode)
        assert isinstance(ast.right, NumberNode)

    def test_parse_binary_subtraction(self):
        """Test parsing subtraction."""
        tokenizer = Tokenizer("10 - 4")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "-"

    def test_parse_binary_multiplication(self):
        """Test parsing multiplication."""
        tokenizer = Tokenizer("6 * 7")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "*"

    def test_parse_binary_division(self):
        """Test parsing division."""
        tokenizer = Tokenizer("15 / 3")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "/"

    def test_parse_power(self):
        """Test parsing exponentiation."""
        tokenizer = Tokenizer("2 ^ 3")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "^"

    def test_parse_unary_minus(self):
        """Test parsing unary minus."""
        tokenizer = Tokenizer("-5")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, UnaryOpNode)
        assert ast.op == "-"
        assert isinstance(ast.operand, NumberNode)

    def test_parse_parentheses(self):
        """Test parsing parenthesized expression."""
        tokenizer = Tokenizer("(3 + 5)")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"

    def test_parse_function(self):
        """Test parsing function call."""
        tokenizer = Tokenizer("sqrt(16)")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        assert isinstance(ast, FunctionNode)
        assert ast.name == "sqrt"
        assert len(ast.args) == 1

    def test_parse_precedence_mul_before_add(self):
        """Test that multiplication has higher precedence than addition."""
        tokenizer = Tokenizer("2 + 3 * 4")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        # Should be parsed as 2 + (3 * 4)
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "*"

    def test_parse_precedence_power_before_mul(self):
        """Test that exponentiation has higher precedence than multiplication."""
        tokenizer = Tokenizer("2 * 3 ^ 2")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        # Should be parsed as 2 * (3 ^ 2)
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "*"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "^"

    def test_parse_right_associative_power(self):
        """Test that exponentiation is right-associative."""
        tokenizer = Tokenizer("2 ^ 3 ^ 2")
        parser = Parser(tokenizer.tokenize())
        ast = parser.parse()
        # Should be parsed as 2 ^ (3 ^ 2)
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "^"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "^"

    def test_parse_empty_expression_raises(self):
        """Test that empty expression raises an error."""
        parser = Parser([])
        with pytest.raises(ValueError, match="Empty expression"):
            parser.parse()


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_evaluate_number(self):
        """Test evaluating a simple number."""
        result = evaluate("42")
        assert result == 42.0

    def test_evaluate_addition(self):
        """Test evaluating addition."""
        result = evaluate("3 + 5")
        assert result == 8.0

    def test_evaluate_subtraction(self):
        """Test evaluating subtraction."""
        result = evaluate("10 - 4")
        assert result == 6.0

    def test_evaluate_multiplication(self):
        """Test evaluating multiplication."""
        result = evaluate("6 * 7")
        assert result == 42.0

    def test_evaluate_division(self):
        """Test evaluating division."""
        result = evaluate("15 / 3")
        assert result == 5.0

    def test_evaluate_power(self):
        """Test evaluating exponentiation."""
        result = evaluate("2 ^ 10")
        assert result == 1024.0

    def test_evaluate_unary_minus(self):
        """Test evaluating unary minus."""
        result = evaluate("-5")
        assert result == -5.0

    def test_evaluate_unary_plus(self):
        """Test evaluating unary plus."""
        result = evaluate("+5")
        assert result == 5.0

    def test_evaluate_complex_expression(self):
        """Test evaluating a complex expression."""
        result = evaluate("3 + 5 * (2 - 8) / 4 ^ 2")
        expected = 3 + 5 * (2 - 8) / 4**2
        assert result == pytest.approx(expected)

    def test_evaluate_right_associative_power(self):
        """Test that power is right-associative: 2^3^2 = 2^9 = 512."""
        result = evaluate("2 ^ 3 ^ 2")
        assert result == 512.0  # 2^(3^2) = 2^9 = 512

    def test_evaluate_nested_parentheses(self):
        """Test evaluating nested parentheses."""
        result = evaluate("((2 + 3) * (4 + 5))")
        assert result == 45.0

    def test_evaluate_division_by_zero_raises(self):
        """Test that division by zero raises an error."""
        with pytest.raises(ValueError, match="Division by zero"):
            evaluate("1 / 0")


# =============================================================================
# Function Evaluation Tests
# =============================================================================


class TestFunctionEvaluation:
    """Tests for mathematical function evaluation."""

    def test_evaluate_sqrt(self):
        """Test square root function."""
        result = evaluate("sqrt(16)")
        assert result == 4.0

    def test_evaluate_sin(self):
        """Test sine function."""
        result = evaluate("sin(0)")
        assert result == pytest.approx(0.0)

    def test_evaluate_cos(self):
        """Test cosine function."""
        result = evaluate("cos(0)")
        assert result == pytest.approx(1.0)

    def test_evaluate_tan(self):
        """Test tangent function."""
        result = evaluate("tan(0)")
        assert result == pytest.approx(0.0)

    def test_evaluate_log(self):
        """Test base-10 logarithm."""
        result = evaluate("log(100)")
        assert result == pytest.approx(2.0)

    def test_evaluate_ln(self):
        """Test natural logarithm."""
        result = evaluate("ln(e)")
        assert result == pytest.approx(1.0)

    def test_evaluate_exp(self):
        """Test exponential function."""
        result = evaluate("exp(0)")
        assert result == pytest.approx(1.0)

    def test_evaluate_abs(self):
        """Test absolute value function."""
        result = evaluate("abs(-5)")
        assert result == 5.0

    def test_evaluate_floor(self):
        """Test floor function."""
        result = evaluate("floor(3.7)")
        assert result == 3.0

    def test_evaluate_ceil(self):
        """Test ceiling function."""
        result = evaluate("ceil(3.2)")
        assert result == 4.0

    def test_evaluate_function_with_expression(self):
        """Test function with expression as argument."""
        result = evaluate("sqrt(9 + 16)")
        assert result == 5.0

    def test_evaluate_nested_functions(self):
        """Test nested function calls."""
        result = evaluate("sqrt(sqrt(16))")
        assert result == 2.0


# =============================================================================
# Visualization Tests
# =============================================================================


class TestVisualization:
    """Tests for the visualization functionality."""

    def test_visualize_simple_addition(self):
        """Test visualization of simple addition."""
        result = visualize_expression("2 + 3")
        lines = result.strip().split("\n")
        # Should have at least the original expression and result
        assert "2 + 3" in lines[0]
        assert "5" in lines[-1]

    def test_visualize_simple_multiplication(self):
        """Test visualization of simple multiplication."""
        result = visualize_expression("3 * 4")
        lines = result.strip().split("\n")
        assert "3 * 4" in lines[0]
        assert "12" in lines[-1]

    def test_visualize_precedence(self):
        """Test that visualization respects precedence."""
        result = visualize_expression("2 + 3 * 4")
        lines = result.strip().split("\n")
        # First line should be original
        assert "2 + 3 * 4" in lines[0]
        # Final line should be result
        assert "14" in lines[-1]
        # Intermediate should show 2 + 12
        intermediate = [line for line in lines if "12" in line and "+" in line]
        assert len(intermediate) >= 1

    def test_visualize_parallel_operations(self):
        """Test that independent operations are shown in parallel."""
        result = visualize_expression("2 * 3 + 4 * 5")
        lines = result.strip().split("\n")
        # First line should have both multiplications
        assert "2 * 3 + 4 * 5" in lines[0]
        # Should have intermediate step with 6 + 20
        intermediate = [line for line in lines if "6" in line and "20" in line]
        assert len(intermediate) >= 1
        # Final should be 26
        assert "26" in lines[-1]

    def test_visualize_parentheses(self):
        """Test visualization with parentheses."""
        result = visualize_expression("(1 + 2) * (3 + 4)")
        lines = result.strip().split("\n")
        assert "(1 + 2) * (3 + 4)" in lines[0]
        # Should show 3 and 7 with * between them at some point (with possible spacing for alignment)
        intermediate = [
            line for line in lines if "3" in line and "*" in line and "7" in line
        ]
        assert len(intermediate) >= 1
        assert "21" in lines[-1]

    def test_visualize_function(self):
        """Test visualization with function."""
        result = visualize_expression("sqrt(16) + 2")
        lines = result.strip().split("\n")
        assert "sqrt(16)" in lines[0]
        assert "6" in lines[-1]

    def test_visualize_contains_connectors(self):
        """Test that visualization contains connector characters."""
        result = visualize_expression("2 + 3 * 4")
        # Should contain box-drawing characters
        assert any(c in result for c in ["│", "└", "┘", "─", "┌", "┐"])

    def test_visualize_power_right_associative(self):
        """Test visualization of right-associative power."""
        result = visualize_expression("2 ^ 3 ^ 2")
        lines = result.strip().split("\n")
        # Should evaluate 3^2 = 9 first, then 2^9 = 512
        intermediate_9 = [
            line for line in lines if "9" in line and "2" in line and "^" in line
        ]
        assert len(intermediate_9) >= 1
        assert "512" in lines[-1]

    def test_visualize_complex_expression(self):
        """Test visualization of the main example expression."""
        result = visualize_expression("3 + 5 * (2 - 8) / 4 ^ 2")
        lines = result.strip().split("\n")
        # Check original expression
        assert "3 + 5 * (2 - 8) / 4 ^ 2" in lines[0]
        # Check final result
        assert "1.125" in lines[-1]


# =============================================================================
# Format Number Tests
# =============================================================================


class TestFormatNumber:
    """Tests for the format_number utility function."""

    def test_format_integer(self):
        """Test formatting an integer."""
        assert format_number(42.0) == "42"

    def test_format_decimal(self):
        """Test formatting a decimal."""
        result = format_number(3.14159)
        assert "3.14159" in result

    def test_format_negative_integer(self):
        """Test formatting a negative integer."""
        assert format_number(-10.0) == "-10"

    def test_format_small_decimal(self):
        """Test formatting a small decimal."""
        result = format_number(0.001)
        assert "0.001" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow_simple(self):
        """Test complete workflow for a simple expression."""
        expr = "2 + 3"
        viz = ExpressionVisualizer(expr)
        result = viz.visualize()
        assert "2 + 3" in result
        assert "5" in result

    def test_full_workflow_complex(self):
        """Test complete workflow for a complex expression."""
        expr = "3 + 5 * (2 - 8) / 4 ^ 2"
        result = evaluate(expr)
        expected = 3 + 5 * (2 - 8) / 4**2
        assert result == pytest.approx(expected)

        viz_result = visualize_expression(expr)
        assert "3 + 5 * (2 - 8) / 4 ^ 2" in viz_result
        assert "1.125" in viz_result

    def test_full_workflow_with_functions(self):
        """Test complete workflow with mathematical functions."""
        expr = "sin(0) + cos(0)"
        result = evaluate(expr)
        assert result == pytest.approx(1.0)

        viz_result = visualize_expression(expr)
        assert "sin(0)" in viz_result
        assert "cos(0)" in viz_result
        assert "1" in viz_result

    def test_show_evaluation_runs(self, capsys):
        """Test that show_evaluation produces output."""
        show_evaluation("2 + 3")
        captured = capsys.readouterr()
        assert "2 + 3" in captured.out
        assert "5" in captured.out

    def test_multiple_parallel_operations(self):
        """Test expression with multiple parallel operations."""
        expr = "(1 + 2) * (3 + 4) + (5 + 6) * (7 + 8)"
        result = evaluate(expr)
        expected = (1 + 2) * (3 + 4) + (5 + 6) * (7 + 8)
        assert result == pytest.approx(expected)

    def test_deeply_nested_expression(self):
        """Test a deeply nested expression."""
        expr = "((((1 + 2))))"
        result = evaluate(expr)
        assert result == 3.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_number(self):
        """Test expression with just a single number."""
        result = evaluate("42")
        assert result == 42.0
        viz = visualize_expression("42")
        assert "42" in viz

    def test_negative_number(self):
        """Test expression with negative number."""
        result = evaluate("-42")
        assert result == -42.0

    def test_double_negative(self):
        """Test double negative."""
        result = evaluate("--5")
        assert result == 5.0

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result1 = evaluate("2+3")
        result2 = evaluate("2 + 3")
        result3 = evaluate("  2   +   3  ")
        assert result1 == result2 == result3 == 5.0

    def test_large_numbers(self):
        """Test with large numbers."""
        result = evaluate("1000000 * 1000000")
        assert result == 1e12

    def test_small_numbers(self):
        """Test with small numbers."""
        result = evaluate("0.000001 * 0.000001")
        assert result == pytest.approx(1e-12)

    def test_pi_constant(self):
        """Test using pi constant."""
        result = evaluate("2 * pi")
        assert result == pytest.approx(2 * math.pi)

    def test_e_constant(self):
        """Test using e constant."""
        result = evaluate("e ^ 2")
        assert result == pytest.approx(math.e**2)

    def test_case_insensitive_functions(self):
        """Test that function names are case-insensitive."""
        result1 = evaluate("sin(0)")
        result2 = evaluate("SIN(0)")
        result3 = evaluate("Sin(0)")
        assert result1 == result2 == result3

    def test_case_insensitive_constants(self):
        """Test that constant names are case-insensitive."""
        result1 = evaluate("pi")
        result2 = evaluate("PI")
        result3 = evaluate("Pi")
        assert result1 == result2 == result3


# =============================================================================
# Compact Mode Tests
# =============================================================================


class TestCompactMode:
    """Tests for the compact mode visualization feature."""

    def test_compact_basic(self):
        """Test basic compact mode removes trailing whitespace."""
        result = visualize_expression("2 + 3", compact=True)
        lines = result.split("\n")
        # No line should have trailing whitespace
        for line in lines:
            assert line == line.rstrip()

    def test_compact_removes_trailing_spaces(self):
        """Test that compact mode removes trailing spaces on all lines."""
        result_compact = visualize_expression("sqrt(16) + 2", compact=True)
        result_normal = visualize_expression("sqrt(16) + 2", compact=False)
        # Compact should be shorter or same length
        assert len(result_compact) <= len(result_normal)
        # No trailing whitespace
        for line in result_compact.split("\n"):
            if line:
                assert line == line.rstrip()

    def test_compact_preserves_token_separation(self):
        """Test that compact mode keeps at least one space between tokens."""
        result = visualize_expression("3 + 5 * (2 - 8) / 4 ^ 2", compact=True)
        lines = result.split("\n")

        # Check that we don't have operators directly touching numbers
        # e.g., "3+-30" should be "3 + -30"
        for line in lines:
            # If line contains operator followed by negative number, should have space
            if "+-" in line or "*-" in line or "/-" in line:
                assert False, f"Missing space before negative number in: {line}"

    def test_compact_maintains_alignment(self):
        """Test that compact mode maintains vertical alignment of connectors."""
        result = visualize_expression("2 + 3 * 4", compact=True)
        lines = result.split("\n")

        # Find position of └ in connector lines and ensure they align
        # with the appropriate expression positions
        expr_line = lines[0]
        assert "2" in expr_line
        assert "+" in expr_line
        assert "3" in expr_line
        assert "*" in expr_line
        assert "4" in expr_line

    def test_compact_function_evaluation(self):
        """Test compact mode with function evaluation."""
        result = visualize_expression("sqrt(16)", compact=True)
        lines = result.split("\n")

        # Result should show 4 somewhere
        assert any("4" in line for line in lines)
        # No trailing whitespace
        for line in lines:
            assert line == line.rstrip()

    def test_compact_vs_non_compact_same_result(self):
        """Test that compact and non-compact produce the same final result."""
        expressions = [
            "2 + 3",
            "3 * 4 + 5",
            "sqrt(16) + 2",
            "(2 + 3) * 4",
        ]
        for expr in expressions:
            result_compact = visualize_expression(expr, compact=True)
            result_normal = visualize_expression(expr, compact=False)

            # Both should show the same final result
            compact_lines = result_compact.strip().split("\n")
            normal_lines = result_normal.strip().split("\n")

            # Last non-empty line should have same numerical value
            compact_result = compact_lines[-1].strip()
            normal_result = normal_lines[-1].strip()
            assert compact_result == normal_result

    def test_compact_with_negative_results(self):
        """Test compact mode correctly handles negative intermediate results."""
        result = visualize_expression("2 - 8", compact=True)
        assert "-6" in result
        # No trailing whitespace
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_compact_default_is_false(self):
        """Test that compact defaults to False."""
        result_default = visualize_expression("2 + 3")
        result_explicit_false = visualize_expression("2 + 3", compact=False)
        assert result_default == result_explicit_false

    def test_compact_complex_expression(self):
        """Test compact mode on a complex expression."""
        result = visualize_expression("sin(pi/2) + cos(0)", compact=True)
        lines = result.split("\n")

        # Should produce correct final result (1 + 1 = 2)
        assert any("2" in line for line in lines)

        # No trailing whitespace
        for line in lines:
            assert line == line.rstrip()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
