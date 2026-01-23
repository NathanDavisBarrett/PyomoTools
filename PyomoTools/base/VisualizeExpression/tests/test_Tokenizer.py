import pytest
from ..Tokenization import Tokenizer, Token, TokenType
import math


class TestTokenizer:
    """Test suite for the Tokenizer class."""

    def test_simple_number(self):
        """Test tokenizing a simple number."""
        tokenizer = Tokenizer("42")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"

    def test_decimal_number(self):
        """Test tokenizing a decimal number."""
        tokenizer = Tokenizer("3.14")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "3.14"

    def test_scientific_notation(self):
        """Test tokenizing scientific notation."""
        tokenizer = Tokenizer("1.5e-10")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "1.5e-10"

    def test_basic_operators(self):
        """Test tokenizing basic arithmetic operators."""
        tokenizer = Tokenizer("+ - * / ^")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 5
        for token in tokens:
            assert token.type == TokenType.OPERATOR
        assert [t.value for t in tokens] == ["+", "-", "*", "/", "^"]

    def test_parentheses(self):
        """Test tokenizing parentheses."""
        tokenizer = Tokenizer("()")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 2
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.RPAREN

    def test_simple_expression(self):
        """Test tokenizing a simple arithmetic expression."""
        tokenizer = Tokenizer("2 + 3 * 4")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 5
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "2"
        assert tokens[1].type == TokenType.OPERATOR
        assert tokens[1].value == "+"
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "3"
        assert tokens[3].type == TokenType.OPERATOR
        assert tokens[3].value == "*"
        assert tokens[4].type == TokenType.NUMBER
        assert tokens[4].value == "4"

    def test_function(self):
        """Test tokenizing a function call."""
        tokenizer = Tokenizer("sin(3.14)")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 4
        assert tokens[0].type == TokenType.FUNCTION
        assert tokens[0].value == "sin"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "3.14"
        assert tokens[3].type == TokenType.RPAREN

    def test_function_with_multiple_args(self):
        """Test tokenizing a function with multiple arguments."""
        tokenizer = Tokenizer("atan(1, 2)")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 6
        assert tokens[0].type == TokenType.FUNCTION
        assert tokens[0].value == "atan"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "1"
        assert tokens[3].type == TokenType.COMMA
        assert tokens[4].type == TokenType.NUMBER
        assert tokens[4].value == "2"
        assert tokens[5].type == TokenType.RPAREN

    def test_nested_functions(self):
        """Test tokenizing nested function calls."""
        tokenizer = Tokenizer("sin(cos(0))")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 7
        assert tokens[0].type == TokenType.FUNCTION
        assert tokens[0].value == "sin"
        assert tokens[2].type == TokenType.FUNCTION
        assert tokens[2].value == "cos"

    def test_relational_operators(self):
        """Test tokenizing relational operators."""
        expressions = [
            ("==", "=="),
            ("<=", "<="),
            (">=", ">="),
            ("<", "<"),
            (">", ">"),
            ("!=", "!="),
        ]
        for expr, expected_op in expressions:
            tokenizer = Tokenizer(expr)
            tokens = tokenizer.tokenize()
            assert len(tokens) == 1
            assert tokens[0].type == TokenType.RELATIONAL
            assert tokens[0].value == expected_op

    def test_relational_expression(self):
        """Test tokenizing a complete relational expression."""
        tokenizer = Tokenizer("3 + 5 == 2 * 4")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 7
        assert tokens[3].type == TokenType.RELATIONAL
        assert tokens[3].value == "=="

    def test_constants(self):
        """Test tokenizing mathematical constants."""
        tokenizer = Tokenizer("pi + e")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 3
        assert tokens[0].type == TokenType.NUMBER
        assert float(tokens[0].value) == pytest.approx(math.pi)
        assert tokens[2].type == TokenType.NUMBER
        assert float(tokens[2].value) == pytest.approx(math.e)

    def test_whitespace_handling(self):
        """Test that whitespace is properly ignored."""
        tokenizer1 = Tokenizer("2+3")
        tokenizer2 = Tokenizer("2 + 3")
        tokenizer3 = Tokenizer("2  +  3")
        tokens1 = tokenizer1.tokenize()
        tokens2 = tokenizer2.tokenize()
        tokens3 = tokenizer3.tokenize()

        assert len(tokens1) == len(tokens2) == len(tokens3) == 3
        for t1, t2, t3 in zip(tokens1, tokens2, tokens3):
            assert t1.value == t2.value == t3.value

    def test_complex_expression(self):
        """Test tokenizing a complex expression."""
        tokenizer = Tokenizer("2 * sin(pi / 4) + sqrt(16) ^ 0.5")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 15

        # Verify key tokens
        assert tokens[0].value == "2"
        assert tokens[1].value == "*"
        assert tokens[2].value == "sin"
        assert tokens[8].value == "+"
        assert tokens[9].value == "sqrt"

    def test_unary_minus(self):
        """Test tokenizing expressions with unary minus."""
        tokenizer = Tokenizer("-5")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 2
        assert tokens[0].type == TokenType.OPERATOR
        assert tokens[0].value == "-"
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "5"

    def test_consecutive_operators(self):
        """Test tokenizing consecutive operators (like in -(-5))."""
        tokenizer = Tokenizer("-(-5)")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 5
        assert tokens[0].value == "-"
        assert tokens[1].value == "("
        assert tokens[2].value == "-"
        assert tokens[3].value == "5"
        assert tokens[4].value == ")"

    def test_case_insensitive_functions(self):
        """Test that function names are case-insensitive."""
        for func_name in ["sin", "SIN", "Sin", "sIn"]:
            tokenizer = Tokenizer(f"{func_name}(0)")
            tokens = tokenizer.tokenize()
            assert tokens[0].type == TokenType.FUNCTION
            assert tokens[0].value == "sin"

    def test_case_insensitive_constants(self):
        """Test that constants are case-insensitive."""
        for const_name in ["pi", "PI", "Pi", "pI"]:
            tokenizer = Tokenizer(const_name)
            tokens = tokenizer.tokenize()
            assert tokens[0].type == TokenType.NUMBER
            assert float(tokens[0].value) == pytest.approx(math.pi)

    def test_position_tracking(self):
        """Test that token positions are tracked correctly."""
        tokenizer = Tokenizer("12 + 34")
        tokens = tokenizer.tokenize()

        assert tokens[0].start == 0
        assert tokens[0].end == 2
        assert tokens[1].start == 3
        assert tokens[1].end == 4
        assert tokens[2].start == 5
        assert tokens[2].end == 7

    def test_unknown_identifier_error(self):
        """Test that unknown identifiers raise an error."""
        tokenizer = Tokenizer("unknown_var")
        with pytest.raises(ValueError, match="Unknown identifier"):
            tokenizer.tokenize()

    def test_unexpected_character_error(self):
        """Test that unexpected characters raise an error."""
        tokenizer = Tokenizer("2 + @")
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenizer.tokenize()

    def test_empty_expression(self):
        """Test tokenizing an empty expression."""
        tokenizer = Tokenizer("")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 0

    def test_whitespace_only_expression(self):
        """Test tokenizing a whitespace-only expression."""
        tokenizer = Tokenizer("   \t  \n  ")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 0

    def test_all_supported_functions(self):
        """Test that all supported functions are recognized."""
        functions = [
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
        ]
        for func in functions:
            tokenizer = Tokenizer(f"{func}(1)")
            tokens = tokenizer.tokenize()
            assert tokens[0].type == TokenType.FUNCTION
            assert tokens[0].value == func

    def test_expression_with_no_spaces(self):
        """Test parsing an expression with no spaces."""
        tokenizer = Tokenizer("2*sin(pi/4)+sqrt(16)^0.5")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 15
        # Just verify it parses without error
        assert all(token.type is not None for token in tokens)
