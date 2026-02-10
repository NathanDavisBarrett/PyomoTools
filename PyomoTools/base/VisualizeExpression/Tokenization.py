from enum import Enum, auto
from dataclasses import dataclass
import math
from typing import List


class TokenType(Enum):
    NUMBER = auto()
    OPERATOR = auto()
    RELATIONAL = auto()  # ==, <=, >=, <, >, !=
    LPAREN = auto()
    RPAREN = auto()
    FUNCTION = auto()
    COMMA = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    start: int
    end: int


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
    RELATIONAL_OPERATORS = {"==", "<=", ">=", "<", ">", "!="}
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
            elif self._try_read_relational():
                pass  # Relational operator was consumed
            elif self._try_read_power_operator():
                pass  # Power operator ** was consumed
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

    def _try_read_relational(self) -> bool:
        """Try to read a relational operator. Returns True if successful."""
        start = self.pos
        char = self.expression[self.pos]

        # Check for Unicode relational operators (single character)
        if char in {"\u2264", "\u2265", "\u2260"}:  # ≤, ≥, ≠
            self.tokens.append(Token(TokenType.RELATIONAL, char, start, start + 1))
            self.pos += 1
            return True

        # Check for two-character operators first
        if self.pos + 1 < len(self.expression):
            two_char = self.expression[self.pos : self.pos + 2]
            if two_char in {"==", "<=", ">=", "!="}:
                self.tokens.append(
                    Token(TokenType.RELATIONAL, two_char, start, start + 2)
                )
                self.pos += 2
                return True

        # Check for single-character operators (< and >)
        if char in {"<", ">"}:
            self.tokens.append(Token(TokenType.RELATIONAL, char, start, start + 1))
            self.pos += 1
            return True

        return False

    def _try_read_power_operator(self) -> bool:
        """Try to read a power operator (**). Returns True if successful."""
        start = self.pos
        if self.pos + 1 < len(self.expression):
            two_char = self.expression[self.pos : self.pos + 2]
            if two_char == "**":
                self.tokens.append(Token(TokenType.OPERATOR, "**", start, start + 2))
                self.pos += 2
                return True
        return False
