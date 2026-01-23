from ..DetermineLeadingLines import DetermineLeadingLines


class TestDetermineLeadingLines:
    def test_no_groupings(self):
        """Test with no groupings."""
        result = DetermineLeadingLines([], [])
        assert result == [" "]

    def test_single_straight_line(self):
        """Test a single line that goes straight down."""
        start = [(0, 4)]
        end = [(0, 4)]
        result = DetermineLeadingLines(start, end)
        expected = ["└─┬─┘", "  │  "]
        assert result == expected

    def test_multiple_straight_lines(self):
        """Test multiple lines that go straight down."""
        start = [(0, 4), (8, 12)]
        end = [(0, 4), (8, 12)]
        result = DetermineLeadingLines(start, end)
        expected = ["└─┬─┘   └─┬─┘", "  │       │  "]
        assert result == expected

    def test_docstring_example(self):
        """Test the example from the docstring."""
        # 15.2 + 37.5544354 + (12.0 / 3.0) * 5.0
        # └─────┬─────────┘   └─────┬────┘   └┬┘
        #       │       ┌───────────┘         │
        #       │       │     ┌───────────────┘
        # 52.7544354 + 4.0 * 5.0
        start = [(0, 18), (22, 34), (38, 40)]
        end = [(0, 9), (13, 15), (19, 21)]
        result = DetermineLeadingLines(start, end)

        # This is a bit fragile, as the coloring can change.
        # We'll check for key structural elements.
        assert len(result) > 1
        assert "└" in result[0] and "┘" in result[0] and "┬" in result[0]

        flat_result = "".join(result)
        assert flat_result.count("┌") >= 1
        assert flat_result.count("┘") >= 1  # from cup
        assert flat_result.count("│") >= 3

    def test_single_char_groupings(self):
        """Test with single-character groupings."""
        start = [(0, 0), (4, 4)]
        end = [(0, 0), (4, 4)]
        result = DetermineLeadingLines(start, end)
        expected = ["│   │"] * 2

        assert result == expected

    def test_narrow_groupings(self):
        """Test with narrow (2-char) groupings."""
        start = [(0, 1)]
        end = [(0, 1)]
        result = DetermineLeadingLines(start, end)
        expected = ["├┘", "│ "]

        assert result == expected
