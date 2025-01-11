import logging
import subprocess
import unittest
import unittest.mock
from os.path import dirname, join
from subprocess import DEVNULL
from typing import NamedTuple
from unittest.mock import Mock

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

APP_PATH = join(dirname(__file__), "..", "src", "saniproclidemo", "cli.py")

APP_BASE_ARGS = ["poetry", "run", "python3", APP_PATH]


class TestCliCommand(unittest.TestCase):
    def test_get_logger_level(self):
        test_cases = [
            (["filter", "unique"], logging.WARNING),
            (["-v", "filter", "unique"], logging.INFO),
            (["-vv", "filter", "unique"], logging.DEBUG),
            (["-vvv", "filter", "unique"], logging.DEBUG),
        ]

        for cli_args, log_level_expected in test_cases:
            with self.subTest(method=cli_args):
                command = CliCommandsDemo(CliArgsNamespaceDemo.from_sys_argv(cli_args))
                self.assertEqual(command.get_logger_level(), log_level_expected)

    def test_error_on_root(self):
        test_cases = [
            ([*APP_BASE_ARGS], 2),
            ([*APP_BASE_ARGS, "filter"], 2),
            ([*APP_BASE_ARGS, "filter", "similar"], 2),
            ([*APP_BASE_ARGS, "filter", "sort-all"], 2),
            ([*APP_BASE_ARGS, "set-operation"], 2),
        ]

        for cli_args, error_code_expected in test_cases:
            with self.subTest(argv=cli_args):
                with self.assertRaises(subprocess.CalledProcessError) as cm:
                    subprocess.run(
                        args=cli_args, stdout=DEVNULL, stderr=DEVNULL, check=True
                    )
                exit_code = cm.exception.returncode
                self.assertEqual(exit_code, error_code_expected)


class TestCliSortAllCommand(unittest.TestCase):
    def test_create_from_cmd(self):
        class namespace(NamedTuple):
            sort_all_method: str
            reverse: bool

        test_cases = [
            (namespace("weight", True), True),
            (namespace("ord-sum", True), True),
        ]

        # just testing --reverse works
        for cli_namespace, predicate in test_cases:
            cls = CliArgsNamespaceDemo()
            cls.sort_all_method = cli_namespace.sort_all_method
            cls.reverse = Mock(cli_namespace.reverse)

            command = CliSortAllCommand.create_from_cmd(cls)
            with self.subTest(method=cli_namespace):
                self.assertEqual(command.command.reverse, predicate)

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

        class namespace(NamedTuple):
            similar_method: str
            kruskal: bool
            prim: bool

        test_cases = [
            (namespace("naive", False, False), NaiveReorderer),
            (namespace("greedy", False, False), GreedyReorderer),
            (namespace("mst", True, False), KruskalMSTReorderer),
            (namespace("mst", False, True), PrimMSTReorderer),
        ]

        for cli_namespace, reorderer_expected in test_cases:
            cls = CliArgsNamespaceDemo()
            cls.similar_method = cli_namespace.similar_method
            cls.kruskal = cli_namespace.kruskal
            cls.prim = cli_namespace.prim
            method = CliSimilarCommand.get_reorderer(cls)

            with self.subTest(method=method):
                self.assertIsInstance(method, reorderer_expected)

    def test_get_reorderer(self):
        test_cases = [
            (["filter", "similar", "naive"], NaiveReorderer),
            (["filter", "similar", "greedy"], GreedyReorderer),
            (["filter", "similar", "mst", "--kruskal"], KruskalMSTReorderer),
            (["filter", "similar", "mst", "--prim"], PrimMSTReorderer),
        ]

        for cli_args, reorderer_expected in test_cases:
            reorderer = CliSimilarCommand.get_reorderer(
                CliArgsNamespaceDemo.from_sys_argv(cli_args)
            )
            with self.subTest(reorderer=reorderer):
                self.assertIsInstance(reorderer, reorderer_expected)
