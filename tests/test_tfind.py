import unittest

from saniproclidemo.tfind import TFindEscaper


class Testformat_a1111compat_token(unittest.TestCase):
    def test_regexp_parentheses(self):
        test_cases = [(r"(", r"\("), (r")", r"\)")]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                self.assertEqual(TFindEscaper.escape_parentheses(input_text), expected)
