import argparse
import functools
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence

from sanipro import common
from sanipro.compatible import Self
from sanipro.filters import (
    exclude,
    fuzzysort,
    mask,
    random,
    reset,
    roundup,
    sort,
    sort_all,
    unique,
)
from sanipro.filters.abc import ReordererStrategy
from sanipro.utils import CommandModuleMap, KeyVal, ModuleMatcher
from saniprocli import cli_hooks
from saniprocli.cli_runner import Runner
from saniprocli.commands import CommandsBase
from saniprocli.help_formatter import SaniproHelpFormatter

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class ExcludeCommand(exclude.ExcludeCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        pass


class SimilarCommand(fuzzysort.SimilarCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        subparser_similar = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Reorders tokens with their similarity.",
            description="Reorders tokens with their similarity.",
        )

        subparser_similar.add_argument(
            "-r",
            "--reverse",
            default=False,
            action="store_true",
            help="With reversed order.",
        )

        subcommand = subparser_similar.add_subparsers(
            title=cls.command_id,
            help="With what method is used to reorder the tokens.",
            description="Reorders tokens with their similarity.",
            dest="similar_method",
            metavar="METHOD",
        )

        subcommand.add_parser(
            MSTAvailable.NAIVE.key,
            formatter_class=SaniproHelpFormatter,
            help=(
                "Calculates all permutations of a sequence of tokens. "
                "Not practical at all."
            ),
        )

        subcommand.add_parser(
            MSTAvailable.GREEDY.key,
            formatter_class=SaniproHelpFormatter,
            help=(
                "Uses a greedy approach that always chooses the next element "
                "with the highest similarity."
            ),
        )

        mst_parser = subcommand.add_parser(
            "mst",
            formatter_class=SaniproHelpFormatter,
            help=("Construct a complete graph with tokens as vertices."),
            description=(
                "Construct a complete graph with tokens as vertices "
                "and similarities as edge weights."
            ),
        )

        mst_group = mst_parser.add_mutually_exclusive_group()

        mst_group.add_argument(
            "-k", "--kruskal", action="store_true", help=("Uses Kruskal's algorithm.")
        )

        mst_group.add_argument(
            "-p", "--prim", action="store_true", help=("Uses Prim's algorithm.")
        )

    @staticmethod
    def get_reorderer(cmd: "DemoCommands") -> ReordererStrategy:
        """Instanciate one reorder function from the parsed result."""

        def get_class(cmd: "DemoCommands"):
            query = cmd.similar_method
            if query != "mst":
                return fuzzysort_apply_from(method=query)

            adapters = [
                [cmd.kruskal, MSTAvailable.KRUSKAL],
                [cmd.prim, MSTAvailable.PRIM],
            ]
            for _flag, _cls in adapters:
                if _flag:
                    # when --kruskal or --prim flag is specified
                    return _cls.val

            _, fallback_cls = adapters[0]
            return fallback_cls.val

        cls = get_class(cmd)
        logger.debug(f"selected module: {cls.__name__}")

        if cls is None:
            raise KeyError

        return cls(strategy=fuzzysort.SequenceMatcherSimilarity())

    @classmethod
    def create_from_cmd(cls, cmd: "DemoCommands", *, reverse=False) -> Self:
        """Alternative method."""
        return cls(reorderer=cls.get_reorderer(cmd), reverse=reverse)


class MSTAvailable(CommandModuleMap):
    NAIVE = KeyVal("naive", fuzzysort.NaiveReorderer)
    GREEDY = KeyVal("greedy", fuzzysort.GreedyReorderer)
    KRUSKAL = KeyVal("kruskal", fuzzysort.KruskalMSTReorderer)
    PRIM = KeyVal("prim", fuzzysort.PrimMSTReorderer)


def fuzzysort_apply_from(*, method: str | None = None) -> type[fuzzysort.MSTReorderer]:
    """
    method を具体的なクラスの名前にマッチングさせる。

    Argument:
        method: コマンドラインで指定された方法.
    """
    # set default
    default = MSTAvailable.GREEDY.key
    if method is None:
        method = default

    mapper = ModuleMatcher(MSTAvailable)
    try:
        return mapper.match(method)
    except KeyError:
        raise ValueError("method name is not found.")


class MaskCommand(mask.MaskCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        subcommand = subparser.add_parser(
            cls.command_id,
            help="Mask tokens with words.",
            description="Mask words specified with another word (optional).",
            formatter_class=SaniproHelpFormatter,
            epilog=(
                (
                    "Note that you can still use the global `--exclude` option"
                    "as well as this filter."
                )
            ),
        )

        subcommand.add_argument("mask", nargs="*", type=str, help="Masks this word.")

        subcommand.add_argument(
            "-t",
            "--replace-to",
            type=str,
            default=r"%%%",
            help="The new character or string replaced to.",
        )


class RandomCommand(random.RandomCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        subcommand = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Shuffles all the prompts altogether.",
            description="Shuffles all the prompts altogether.",
        )

        subcommand.add_argument(
            "-b",
            "--seed",
            default=None,
            type=int,
            help="Fixed randomness to this value.",
        )


class ResetCommand(reset.ResetCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        subcommand = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Initializes all the weight of the tokens.",
            description="Initializes all the weight of the tokens.",
        )

        subcommand.add_argument(
            "-v",
            "--value",
            default=1.0,
            type=float,
            help="Fixes the weight to this value.",
        )


class RoundUpCommand(roundup.RoundUpCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction): ...


class SortCommand(sort.SortCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        subcommand = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Reorders duplicate tokens.",
            description="Reorders duplicate tokens.",
            epilog="This command reorders tokens with their weights by default.",
        )

        subcommand.add_argument(
            "-r", "--reverse", action="store_true", help="With reversed order."
        )


def sort_all_apply_from(*, method: str | None = None) -> functools.partial:
    """
    method を具体的なクラスの名前にマッチングさせる。

    Argument:
        method: コマンドラインで指定された方法.
    """
    # set default
    default = sort_all.Available.LEXICOGRAPHICAL.key
    if method is None:
        method = default

    mapper = ModuleMatcher(sort_all.Available)
    try:
        partial = functools.partial(sorted, key=mapper.match(method))
        return partial
    except KeyError:
        raise ValueError("method name is not found.")


class SortAllCommand(sort_all.SortAllCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        sort_all_subparser = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Reorders all the prompts.",
            description="Reorders all the prompts.",
        )

        sort_all_subparser.add_argument(
            "-r", "--reverse", action="store_true", help="With reversed order."
        )

        subcommand = sort_all_subparser.add_subparsers(
            title=cls.command_id,
            description="The available method to sort the tokens.",
            dest="sort_all_method",
            metavar="METHOD",
        )

        subcommand.add_parser(
            sort_all.Available.LEXICOGRAPHICAL.key,
            help="Sort the prompt with lexicographical order. Familiar sort method.",
        )

        subcommand.add_parser(
            sort_all.Available.LENGTH.key,
            help=(
                "Reorder the token length."
                "This behaves slightly similar as 'ord-sum' method."
            ),
        )

        subcommand.add_parser(
            sort_all.Available.STRENGTH.key, help="Reorder the tokens by their weights."
        )

        subcommand.add_parser(
            sort_all.Available.ORD_SUM.key,
            help=(
                "Reorder the tokens by its sum of character codes."
                "This behaves slightly similar as 'length' method."
            ),
        )

    @classmethod
    def create_from_cmd(cls, cmd: "DemoCommands", *, reverse=False) -> Self:
        """Alternative method."""
        partial = sort_all_apply_from(method=cmd.sort_all_method)
        return cls(partial, reverse=reverse)


class UniqueCommand(unique.UniqueCommand):
    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction):
        subparser_unique = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Removes duplicated tokens, and uniquify them.",
            description="Removes duplicated tokens, and uniquify them.",
            epilog="",
        )

        subparser_unique.add_argument(
            "-r",
            "--reverse",
            action="store_true",
            help="Make the token with the heaviest weight survived.",
        )


class Command(ABC):
    @classmethod
    @abstractmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction): ...


class DemoCommands(CommandsBase):
    """Custom subcommand implementation by user"""

    # global options
    exclude: Sequence[str]
    roundup = 2
    replace_to = ""
    mask: Sequence[str]
    use_parser_v2 = False

    # subcommands options
    reverse = False
    seed: int | None = None
    value: float | None = None
    similar_method = None
    sort_all_method = None
    kruskal = None
    prim = None

    command_classes = (
        MaskCommand,
        RandomCommand,
        ResetCommand,
        SimilarCommand,
        SortAllCommand,
        SortCommand,
        UniqueCommand,
    )

    @classmethod
    def append_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-u",
            "--roundup",
            default=cls.roundup,
            type=int,
            help=(
                "All the token with weights (x > 1.0 or x < 1.0) "
                "will be rounded up to n digit(s)."
            ),
        )

        parser.add_argument(
            "-x",
            "--exclude",
            type=str,
            nargs="*",
            help=(
                "Exclude this token from the original prompt. "
                "Multiple options can be specified."
            ),
        )

        parser.add_argument(
            "--use-parser-v2",
            "-2",
            action="store_true",
            help=(
                "Switch to use another version of the parser instead. "
                "This might be inferrior to the default parser "
                "as it only parses the prompt and does nothing at all."
            ),
        )

    @classmethod
    def append_subparser(cls, parser: argparse.ArgumentParser) -> None:
        subparser = parser.add_subparsers(
            title="filter",
            description=(
                "List of available filters that can be applied to the prompt. "
                "Just one filter can be applied at once."
            ),
            dest="filter",
            metavar="FILTER",
        )

        for cmd in cls.command_classes:
            cmd.inject_subparser(subparser)

    def _get_pipeline_from(self, use_parser_v2: bool) -> common.PromptPipeline:
        delimiter = common.Delimiter(self.input_delimiter, self.output_delimiter)
        pipeline = None
        if not use_parser_v2:
            pipeline = delimiter.create_pipeline(common.PromptPipelineV1)
        else:
            pipeline = delimiter.create_pipeline(common.PromptPipelineV2)
        return pipeline

    def get_pipeline(self) -> common.PromptPipeline:
        """This is a pipeline for the purpose of showcasing.
        Since all the parameters of each command is variable, the command
        sacrifies the composability.
        It is good for you to create your own pipeline, and name it
        so you can use it as a preset."""

        command_ids = [cmd.command_id for cmd in self.command_classes]
        command_funcs = (
            lambda: MaskCommand(self.mask, self.replace_to),
            lambda: RandomCommand(self.seed),
            lambda: ResetCommand(self.value),
            lambda: SimilarCommand.create_from_cmd(cmd=self, reverse=self.reverse),
            lambda: SortAllCommand.create_from_cmd(cmd=self, reverse=self.reverse),
            lambda: SortCommand(self.reverse),
            lambda: UniqueCommand(self.reverse),
        )
        command_map = dict(zip(command_ids, command_funcs, strict=True))

        if self.use_parser_v2:
            if self.filter in command_ids:
                raise NotImplementedError(
                    f"the '{self.filter}' command is not available "
                    "when using parse_v2."
                )

            logger.warning("using parser_v2.")

        pipeline = self._get_pipeline_from(self.use_parser_v2)

        # always round
        pipeline.append_command(RoundUpCommand(self.roundup))

        if self.filter is not None:
            lambd = command_map[self.filter]
            pipeline.append_command(lambd())

        if self.exclude:
            pipeline.append_command(ExcludeCommand(self.exclude))

        return pipeline


def app():
    try:
        args = DemoCommands.from_sys_argv(sys.argv[1:])
        cli_hooks.execute(cli_hooks.init)

        log_level = args.get_logger_level()
        logger_root.setLevel(log_level)

        args.debug()
        runner = Runner.from_args(args)
        runner.run()
    except Exception as e:
        logger.exception(f"error: {e}")
    finally:
        sys.exit(1)


if __name__ == "__main__":
    app()
