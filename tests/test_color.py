import re
import unittest

from saniprocli.color import EscSeqWrapper


class TestEscSeqWrapper(unittest.TestCase):
    def test_regex(self):
        """test if the regex is correct."""

        test_cases = [
            "\33[31m",
            "\33[31;1m",
            "\33[31;1;4m",
            "\33[31mHello\33[0m",
            "\33[31;1mHello\33[0m",
            "\33[31;1;4mHello\33[0m",
        ]

        for input_text in test_cases:
            with self.subTest(pattern=input_text):
                self.assertIsNotNone(re.match(EscSeqWrapper.pattern, input_text))

    def test_wrap(self):
        test_cases = [
            ("\33[31mfoo\33[0m", "\1\33[31m\2foo\1\33[0m\2"),
            ("\33[31;1mfoo\33[0m", "\1\33[31;1m\2foo\1\33[0m\2"),
            ("\33[31;1;4mfoo\33[0m", "\1\33[31;1;4m\2foo\1\33[0m\2"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(pattern=input_text):
                self.assertEqual(EscSeqWrapper.wrap(input_text), expected)
