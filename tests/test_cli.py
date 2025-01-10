import unittest
from dataclasses import dataclass

from saniproclidemo.cli import (
    CliArgsNamespaceDemo,
    CliSimilarCommand,
    GreedyReorderer,
    KruskalMSTReorderer,
    NaiveReorderer,
    PrimMSTReorderer,
)


@dataclass
class NamespaceMock:
    similar_method: str
    kruskal: bool
    prim: bool


class TestCliSimilarCommand(unittest.TestCase):
    def test_get_class(self):
        test_cases = [
            (NamespaceMock("naive", False, False), NaiveReorderer),
            (NamespaceMock("greedy", False, False), GreedyReorderer),
            (NamespaceMock("mst", True, False), KruskalMSTReorderer),
            (NamespaceMock("mst", False, True), PrimMSTReorderer),
        ]

        for args, expect in test_cases:
            reorderer = CliSimilarCommand.get_reorderer(args)
            with self.subTest(reorderer=reorderer):
                self.assertIsInstance(reorderer, expect)

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


class TestCliArgsNamespaceDemo(unittest.TestCase):
    def setUp(self) -> None:
        pass
