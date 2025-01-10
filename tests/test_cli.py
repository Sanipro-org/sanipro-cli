import logging
import unittest
from dataclasses import dataclass

from sanipro.filters.utils import (
    sort_by_length,
    sort_by_ord_sum,
    sort_by_weight,
    sort_lexicographically,
)

from saniproclidemo.cli import (
    CliArgsNamespaceDemo,
    CliCommandsDemo,
    CliSimilarCommand,
    CliSortAllCommand,
    GreedyReorderer,
    KruskalMSTReorderer,
    NaiveReorderer,
    PrimMSTReorderer,
)


class TestCliCommand(unittest.TestCase):
    def test_get_logger_level(self):
        test_cases = [
            (["filter", "unique"], logging.WARNING),
            (["-v", "filter", "unique"], logging.INFO),
            (["-vv", "filter", "unique"], logging.DEBUG),
            (["-vvv", "filter", "unique"], logging.DEBUG),
        ]

        for args, expect in test_cases:
            with self.subTest(method=args):
                command = CliCommandsDemo(CliArgsNamespaceDemo.from_sys_argv(args))
                self.assertEqual(command.get_logger_level(), expect)


@dataclass
class NamespaceMock:
    similar_method: str
    kruskal: bool
    prim: bool


class TestCliSortAllCommand(unittest.TestCase):
    @unittest.skip("skip")
    def test_create_from_cmd(self):
        test_cases = [
            (["filter", "sort-all", "lexicographical"], sort_lexicographically),
            (["filter", "sort-all", "length"], sort_by_length),
            (["filter", "sort-all", "weight"], sort_by_weight),
            (["filter", "sort-all", "ord-sum"], sort_by_ord_sum),
        ]

        for args, expect in test_cases:
            method = CliSortAllCommand.create_from_cmd(
                CliArgsNamespaceDemo.from_sys_argv(args)
            )
            with self.subTest(method=method):
                self.assertEqual(method, expect)

    def test_query_strategy(self):
        test_cases = [
            ("lexicographical", sort_lexicographically),
            ("length", sort_by_length),
            ("weight", sort_by_weight),
            ("ord-sum", sort_by_ord_sum),
        ]

        for input_text, expect in test_cases:
            method = CliSortAllCommand._query_strategy(input_text)
            with self.subTest(method=method):
                self.assertEqual(method, expect)


class TestCliSimilarCommand(unittest.TestCase):
    def test_get_class(self):
        test_cases = [
            (NamespaceMock("naive", False, False), NaiveReorderer),
            (NamespaceMock("greedy", False, False), GreedyReorderer),
            (NamespaceMock("mst", True, False), KruskalMSTReorderer),
            (NamespaceMock("mst", False, True), PrimMSTReorderer),
        ]

        for args, expect in test_cases:
            method = CliSimilarCommand.get_reorderer(args)
            with self.subTest(method=method):
                self.assertIsInstance(method, expect)

    def test_get_reorderer(self):
        test_cases = [
            (["filter", "similar", "naive"], NaiveReorderer),
            (["filter", "similar", "greedy"], GreedyReorderer),
            (["filter", "similar", "mst", "--kruskal"], KruskalMSTReorderer),
            (["filter", "similar", "mst", "--prim"], PrimMSTReorderer),
        ]

        for args, expect in test_cases:
            reorderer = CliSimilarCommand.get_reorderer(
                CliArgsNamespaceDemo.from_sys_argv(args)
            )
            with self.subTest(reorderer=reorderer):
                self.assertIsInstance(reorderer, expect)
