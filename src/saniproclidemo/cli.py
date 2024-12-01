import argparse
import atexit
import functools
import logging
import os
import pprint
import readline
import sys
import typing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NamedTuple

from sanipro import pipeline
from sanipro.abc import MutablePrompt, Prompt
from sanipro.compatible import Self
from sanipro.filters.abc import Command, ReordererStrategy
from sanipro.filters.exclude import ExcludeCommand
from sanipro.filters.fuzzysort import (
    GreedyReorderer,
    KruskalMSTReorderer,
    NaiveReorderer,
    PrimMSTReorderer,
    ReordererStrategy,
    SequenceMatcherSimilarity,
    SimilarCommand,
)
from sanipro.filters.mask import MaskCommand
from sanipro.filters.random import RandomCommand
from sanipro.filters.reset import ResetCommand
from sanipro.filters.roundup import RoundUpCommand
from sanipro.filters.sort import SortCommand
from sanipro.filters.sort_all import SortAllCommand
from sanipro.filters.unique import UniqueCommand
from sanipro.filters.utils import (
    sort_by_length,
    sort_by_ord_sum,
    sort_by_weight,
    sort_lexicographically,
)
from sanipro.modules import create_pipeline

from saniprocli import cli_hooks, color
from saniprocli.color import style
from saniprocli.commands import CommandsBase
from saniprocli.help_formatter import SaniproHelpFormatter
from saniprocli.logger import logger_fp

logging.basicConfig(
    format=(
        style("[%(levelname)s] %(module)s/%(funcName)s (%(lineno)d):") + " %(message)s"
    ),
    datefmt=r"%Y-%m-%d %H:%M:%S",
)

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class CmdModuleTuple(NamedTuple):
    key: str
    callable_name: typing.Any


class ModuleMapper:
    """The name definition for the subcommands."""

    @classmethod
    def list_commands(cls) -> dict:
        """Get a list of available subcommand representations"""
        return dict(cls.__dict__[var] for var in vars(cls) if var.isupper())


class ModuleMatcher:
    def __init__(self, commands: type[ModuleMapper]):
        if not issubclass(commands, ModuleMapper):
            raise TypeError("invalid command module map was given!")
        self.commands = commands

    def match(self, method: str) -> typing.Any:
        # Todo: typing.Any を利用しないいい方法はないか？
        f = self.commands.list_commands()
        logger.debug(f"{method=}")
        pprint.pprint(f, logger_fp)

        result = f.get(method)
        if result is None:
            raise KeyError
        return result


class CliCommandInterface(ABC):
    @abstractmethod
    def __init__(self, command: Command) -> None: ...

    @abstractmethod
    def execute(self, prompt: Prompt) -> MutablePrompt: ...

    @classmethod
    @abstractmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None: ...


class CliCommandBase(CliCommandInterface):
    command_id: str
    command: Command

    def __init__(self, command: Command) -> None:
        self.command = command

    def execute(self, prompt: Prompt) -> MutablePrompt:
        return self.command.execute(prompt)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        pass


class CliExcludeCommand(CliCommandBase):
    command_id: str = "exclude"


class SimilarModuleMapper(ModuleMapper):
    NAIVE = CmdModuleTuple("naive", NaiveReorderer)
    GREEDY = CmdModuleTuple("greedy", GreedyReorderer)
    KRUSKAL = CmdModuleTuple("kruskal", KruskalMSTReorderer)
    PRIM = CmdModuleTuple("prim", PrimMSTReorderer)


class CliSimilarCommand(CliCommandBase):
    command_id: str = "similar"

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
            SimilarModuleMapper.NAIVE.key,
            formatter_class=SaniproHelpFormatter,
            help=(
                "Calculates all permutations of a sequence of tokens. "
                "Not practical at all."
            ),
        )

        subcommand.add_parser(
            SimilarModuleMapper.GREEDY.key,
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

    @classmethod
    def query_strategy(
        cls, method: str | None = None
    ) -> type[ReordererStrategy] | None:
        """Matches the methods specified on the command line
        to the names of concrete classes.
        Searches other than what the strategy uses MST."""

        default = SimilarModuleMapper.GREEDY.key
        if method is None:
            method = default

        mapper = ModuleMatcher(SimilarModuleMapper)
        matched = mapper.match(method)

        if issubclass(matched, ReordererStrategy):
            return matched
        return None

    @classmethod
    def get_reorderer(cls, cmd: "DemoCommands") -> ReordererStrategy:
        """Instanciate one reorder function from the parsed result."""

        def get_class(cmd: "DemoCommands") -> type[ReordererStrategy] | None:
            query = cmd.similar_method
            if query != "mst":
                return cls.query_strategy(method=query)
            else:
                adapters = [
                    [cmd.kruskal, SimilarModuleMapper.KRUSKAL],
                    [cmd.prim, SimilarModuleMapper.PRIM],
                ]
                for _flag, _cls in adapters:
                    if _flag and isinstance(_cls, CmdModuleTuple):
                        # when --kruskal or --prim flag is specified
                        return _cls.callable_name

                _, fallback_cls = adapters[0]
                if isinstance(fallback_cls, CmdModuleTuple):
                    return fallback_cls.callable_name
            return None

        selected_cls = get_class(cmd)
        if selected_cls is not None:
            if issubclass(selected_cls, ReordererStrategy):
                logger.debug(f"selected module: {selected_cls.__name__}")
                return selected_cls(strategy=SequenceMatcherSimilarity())

        raise ValueError("failed to find reorder function.")

    @classmethod
    def create_from_cmd(cls, cmd: "DemoCommands", *, reverse=False) -> Self:
        """Alternative method."""
        return SimilarCommand(reorderer=cls.get_reorderer(cmd), reverse=reverse)


class CliMaskCommand(CliCommandBase):
    command_id: str = "mask"

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


class CliRandomCommand(CliCommandBase):
    command_id: str = "random"

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


class CliResetCommand(CliCommandBase):
    command_id: str = "reset"

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


class CliRoundUpCommand(CliCommandBase):
    command_id: str = "roundup"


class CliSortCommand(CliCommandBase):
    command_id: str = "sort"

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


class SortAllModuleMapper(ModuleMapper):
    LEXICOGRAPHICAL = CmdModuleTuple("lexicographical", sort_lexicographically)
    LENGTH = CmdModuleTuple("length", sort_by_length)
    STRENGTH = CmdModuleTuple("weight", sort_by_weight)
    ORD_SUM = CmdModuleTuple("ord-sum", sort_by_ord_sum)


class CliSortAllCommand(CliCommandBase):
    command_id: str = "sort-all"

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
            SortAllModuleMapper.LEXICOGRAPHICAL.key,
            help="Sort the prompt with lexicographical order. Familiar sort method.",
        )

        subcommand.add_parser(
            SortAllModuleMapper.LENGTH.key,
            help=(
                "Reorder the token length."
                "This behaves slightly similar as 'ord-sum' method."
            ),
        )

        subcommand.add_parser(
            SortAllModuleMapper.STRENGTH.key,
            help="Reorder the tokens by their weights.",
        )

        subcommand.add_parser(
            SortAllModuleMapper.ORD_SUM.key,
            help=(
                "Reorder the tokens by its sum of character codes."
                "This behaves slightly similar as 'length' method."
            ),
        )

    @classmethod
    def query_strategy(cls, *, method: str | None = None) -> functools.partial:
        """
        method を具体的なクラスの名前にマッチングさせる。

        Argument:
            method: コマンドラインで指定された方法.
        """
        # set default
        default = SortAllModuleMapper.LEXICOGRAPHICAL.key
        if method is None:
            method = default

        mapper = ModuleMatcher(SortAllModuleMapper)
        try:
            partial = functools.partial(sorted, key=mapper.match(method))
            return partial
        except KeyError:
            raise ValueError("method name is not found.")

    @classmethod
    def create_from_cmd(cls, cmd: "DemoCommands", *, reverse=False) -> Self:
        """Alternative method."""

        partial = cls.query_strategy(method=cmd.sort_all_method)
        return SortAllCommand(partial, reverse=reverse)


class CliUniqueCommand(CliCommandBase):
    command_id: str = "unique"

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
    similar_method: str | None = None
    sort_all_method = None
    kruskal = None
    prim = None

    command_classes = (
        CliMaskCommand,
        CliRandomCommand,
        CliResetCommand,
        CliSimilarCommand,
        CliSortAllCommand,
        CliSortCommand,
        CliUniqueCommand,
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

    def _get_pipeline_from(self, use_parser_v2: bool) -> pipeline.PromptPipeline:
        delimiter = pipeline.Delimiter(self.input_delimiter, self.output_delimiter)
        pipe = None
        if not use_parser_v2:
            pipe = create_pipeline(delimiter, pipeline.PromptPipelineV1)
        else:
            pipe = create_pipeline(delimiter, pipeline.PromptPipelineV2)
        return pipe

    def get_pipeline(self) -> pipeline.PromptPipeline:
        """This is a pipeline for the purpose of showcasing.
        Since all the parameters of each command is variable, the command
        sacrifies the composability.
        It is good for you to create your own pipeline, and name it
        so you can use it as a preset."""

        command_ids = [cmd.command_id for cmd in self.command_classes]
        command_funcs = (
            lambda: CliMaskCommand(MaskCommand(self.mask, self.replace_to)),
            lambda: CliRandomCommand(RandomCommand(self.seed)),
            lambda: CliResetCommand(ResetCommand(self.value)),
            lambda: CliSimilarCommand.create_from_cmd(cmd=self, reverse=self.reverse),
            lambda: CliSortAllCommand.create_from_cmd(cmd=self, reverse=self.reverse),
            lambda: CliSortCommand(SortCommand(self.reverse)),
            lambda: CliUniqueCommand(UniqueCommand(self.reverse)),
        )
        command_map = dict(zip(command_ids, command_funcs, strict=True))

        if self.use_parser_v2:
            if self.filter in command_ids:
                raise NotImplementedError(
                    f"the '{self.filter}' command is not available "
                    "when using parse_v2."
                )

            logger.warning("using parser_v2.")

        pipe = self._get_pipeline_from(self.use_parser_v2)

        # always round
        pipe.append_command(CliRoundUpCommand(RoundUpCommand(self.roundup)))

        if self.filter is not None:
            lambd = command_map[self.filter]
            pipe.append_command(lambd())

        if self.exclude:
            pipe.append_command(CliExcludeCommand(ExcludeCommand(self.exclude)))

        return pipe


def prepare_readline() -> None:
    histfile = os.path.join(os.path.expanduser("~"), ".sanipro_history")

    try:
        readline.read_history_file(histfile)
        h_len = readline.get_current_history_length()
    except FileNotFoundError:
        open(histfile, "wb").close()
        h_len = 0

    def save(prev_h_len, histfile):
        new_h_len = readline.get_current_history_length()
        readline.append_history_file(new_h_len - prev_h_len, histfile)

    atexit.register(save, h_len, histfile)


def app():
    try:
        args = DemoCommands.from_sys_argv(sys.argv[1:])
        cli_hooks.on_init.append(prepare_readline)
        cli_hooks.execute(cli_hooks.on_init)

        log_level = args.get_logger_level()
        logger_root.setLevel(log_level)

        args.debug()
        runner = args.to_runner()
        runner.run()
    except Exception as e:
        logger.exception(f"error: {e}")
    finally:
        sys.exit(1)


if __name__ == "__main__":
    app()
