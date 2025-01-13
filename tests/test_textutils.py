import unittest

from saniprocli.textutils import CSVUtilsBase


class TestCSVUtils(unittest.TestCase):
    def setUp(self):
        class CSVUtilsTest(CSVUtilsBase):
            def _do_preprocess(self, column: list[str]) -> list[str]:
                return column

        self.csvutil = CSVUtilsTest

    def test_prepare_kv(self):
        test_cases = [
            (
                ["line1,line2", "line3,line4", "line5,line6"],
                (1, 2),
                {"line1": "line2", "line3": "line4", "line5": "line6"},
            ),
            (
                ["line1,line2,line3", "line4,line5,line6", "line7,line8,line9"],
                (1, 3),
                {"line1": "line3", "line4": "line6", "line7": "line9"},
            ),
        ]

        for input_text, rownum, expected in test_cases:
            csvutil = self.csvutil(input_text, ",")
            self.assertEqual(csvutil.prepare_kv(*rownum), expected)

    def test_is_ranged_or_raise(self):
        with self.assertRaises(ValueError):
            self.csvutil.is_ranged_or_raise(0, 1)

    def test_is_different_idx_or_raise(self):
        with self.assertRaises(ValueError):
            self.csvutil.is_different_idx_or_raise(1, 1)

    def test_index_error(self):
        rowdata = ["line1,line2", "line4,line5", "line6,line7"]
        csvutil = self.csvutil(rowdata, ",")

        with self.assertRaises(IndexError):
            csvutil.prepare_kv(1, 3)
