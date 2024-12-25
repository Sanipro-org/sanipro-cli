import argparse
import atexit
import functools
import logging
import os
import pprint
import readline
import subprocess
import sys
import tempfile
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple

from sanipro.abc import MutablePrompt, Prompt
from sanipro.compatible import Self
from sanipro.delimiter import Delimiter
from sanipro.diff import PromptDifferenceDetector
from sanipro.filter_exec import FilterExecutor
from sanipro.filters.abc import ExecutePrompt, ReordererStrategy
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
from sanipro.parser import (
    DummyParser,
    ParserInterface,
    ParserV1,
    ParserV2,
    TokenInteractive,
    TokenNonInteractive,
)
from sanipro.pipeline import PromptPipeline, PromptPipelineV1, PromptPipelineV2
from sanipro.promptset import SetCalculatorWrapper
from sanipro.tokenizer import PromptTokenizer, PromptTokenizerV1, PromptTokenizerV2

from saniprocli import cli_hooks, inputs
from saniprocli.abc import CliRunnable, InputStrategy, StatShowable
from saniprocli.cli_runner import (
    ExecuteDual,
    ExecuteSingle,
    RunnerInteractive,
    RunnerNonInteractive,
)
from saniprocli.color import style
from saniprocli.commands import CliArgsNamespaceDefault, CliCommands
from saniprocli.help_formatter import SaniproHelpFormatter
from saniprocli.logger import logger_fp
from saniprocli.sanipro_argparse import SaniproArgumentParser

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
        # TODO: better type hinting without using typing.Any.
        f = self.commands.list_commands()
        logger.debug(f"{method=}")
        pprint.pprint(f, logger_fp)

        result = f.get(method)
        if result is None:
            raise KeyError
        return result


class SubparserInjectable(ABC):
    """The trait with the ability to inject a subparser."""

    @classmethod
    @abstractmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        """Injects subparser."""


class CliCommand(ExecutePrompt, SubparserInjectable):
    """The wrapper class for the filter commands
    with the addition of subparser."""

    command_id: str
    command: ExecutePrompt

    def __init__(self, command: ExecutePrompt) -> None:
        self.command = command

    def execute_prompt(self, prompt: Prompt) -> MutablePrompt:
        return self.command.execute_prompt(prompt)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        """Does nothing by default."""


class CliExcludeCommand(CliCommand):
    command_id: str = "exclude"

    def __init__(self, excludes: Sequence[str]):
        super().__init__(ExcludeCommand(excludes))


class SimilarModuleMapper(ModuleMapper):
    NAIVE = CmdModuleTuple("naive", NaiveReorderer)
    GREEDY = CmdModuleTuple("greedy", GreedyReorderer)
    KRUSKAL = CmdModuleTuple("kruskal", KruskalMSTReorderer)
    PRIM = CmdModuleTuple("prim", PrimMSTReorderer)


class CliSimilarCommand(CliCommand):
    command_id: str = "similar"

    def __init__(self, reorderer: ReordererStrategy, *, reverse=False):
        super().__init__(SimilarCommand(reorderer, reverse=reverse))

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
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

        cls._add_subcommands(subcommand)

    @classmethod
    def _add_subcommands(cls, subcommand: argparse._SubParsersAction) -> None:
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
    def _query_strategy(
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
    def get_reorderer(cls, cmd: "CliArgsNamespaceDemo") -> ReordererStrategy:
        """Instanciate one reorder function from the parsed result."""

        def get_class(cmd: "CliArgsNamespaceDemo") -> type[ReordererStrategy] | None:
            query = cmd.similar_method
            if query != "mst":
                return cls._query_strategy(method=query)
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
    def create_from_cmd(cls, cmd: "CliArgsNamespaceDemo", *, reverse=False) -> Self:
        """Alternative method."""

        return cls(reorderer=cls.get_reorderer(cmd), reverse=reverse)


class CliMaskCommand(CliCommand):
    command_id: str = "mask"

    def __init__(self, excludes: Sequence[str], replace_to: str):
        self.command = MaskCommand(excludes, replace_to)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
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


class CliRandomCommand(CliCommand):
    command_id: str = "random"

    def __init__(self, seed: int | None = None):
        self.command = RandomCommand(seed)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
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


class CliResetCommand(CliCommand):
    command_id: str = "reset"

    def __init__(self, new_value: float | None = None) -> None:
        self.command = ResetCommand(new_value)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
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


class CliRoundUpCommand(CliCommand):
    command_id: str = "roundup"

    def __init__(self, digits: int):
        self.command = RoundUpCommand(digits)


class CliSortCommand(CliCommand):
    command_id: str = "sort"

    def __init__(self, reverse: bool = False):
        self.command = SortCommand(reverse)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
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


class CliSortAllCommand(CliCommand):
    command_id: str = "sort-all"

    def __init__(self, sorted_partial: partial, reverse: bool = False):
        self.command = SortAllCommand(sorted_partial, reverse)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Reorders all the prompts.",
            description="Reorders all the prompts.",
        )

        parser.add_argument(
            "-r", "--reverse", action="store_true", help="With reversed order."
        )

        subcommand = parser.add_subparsers(
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
    def _query_strategy(cls, *, method: str | None = None) -> functools.partial:
        """Matches `method` to the name of a concrete class."""
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
    def create_from_cmd(cls, cmd: "CliArgsNamespaceDemo", *, reverse=False) -> Self:
        """Alternative method."""

        partial = cls._query_strategy(method=cmd.sort_all_method)
        return cls(partial, reverse=reverse)


class CliUniqueCommand(CliCommand):
    command_id: str = "unique"

    def __init__(self, reverse: bool = False):
        self.command = UniqueCommand(reverse)

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help="Removes duplicated tokens, and uniquify them.",
            description="Removes duplicated tokens, and uniquify them.",
        )

        parser.add_argument(
            "-r",
            "--reverse",
            action="store_true",
            help="Make the token with the heaviest weight survived.",
        )


class CliSubcommandSearchTag(SubparserInjectable):
    command_id: str = "tfind"

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help=("Outputs the number of posts corresponds the tag specified."),
            description=(
                "In this mode a user specifies a CSV file acting as a key-value storage. "
                "The first which is the `key` column corresponds to the name of the tag, "
                "so do second which is the `value` column to the count of the assigned tag."
            ),
        )

        parser.add_argument(
            "infile",
            type=argparse.FileType("r"),
            help=(
                "Specifies the text file comprised from two columns, "
                "each separated with delimiter."
            ),
        )

        parser.add_argument(
            "-i",
            "--interactive",
            default=False,
            action="store_true",
            help=(
                "Provides the REPL interface to play with prompts. "
                "The program behaves like the Python interpreter."
            ),
        )

        parser.add_argument(
            "-k",
            "--key-field",
            default=1,
            type=int,
            help="Select this field number's element as a key.",
        )

        parser.add_argument(
            "-v",
            "--value-field",
            default=2,
            type=int,
            help="Select this field number's element as a value.",
        )

        parser.add_argument(
            "-d",
            "--field-delimiter",
            default=",",
            type=str,
            help="Use this character as a field separator.",
        )

        parser.add_argument(
            "-t",
            "--tempdir",
            default="/dev/shm",
            type=str,
            help=(
                "Use this directory as a tempfile storage. Technically speaking, "
                "the program creates a new text file by extracting the field "
                "from a csv file, format and save them so the GNU Readline "
                "can read the histfile on this directory."
            ),
        )

        parser.add_argument(
            "-c",
            "--clipboard",
            default=False,
            action="store_true",
            help="Copy the result to the clipboard if possible.",
        )


class CliSubcommandFilter(SubparserInjectable):
    command_id: str = "filter"

    filter_classes = (
        CliMaskCommand,
        CliRandomCommand,
        CliResetCommand,
        CliSimilarCommand,
        CliSortAllCommand,
        CliSortCommand,
        CliUniqueCommand,
    )

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description=("Applies a filter to the prompt."),
            help=("Applies a filter to the prompt."),
        )

        parser.add_argument(
            "-i",
            "--interactive",
            default=False,
            action="store_true",
            help=(
                "Provides the REPL interface to play with prompts. "
                "The program behaves like the Python interpreter."
            ),
        )

        subparser = parser.add_subparsers(
            title="filter",
            description=(
                "List of available filters that can be applied to the prompt. "
                "Just one filter can be applied at once."
            ),
            dest="filter_id",
            metavar="FILTER",
            required=True,
        )

        for cmd in cls.filter_classes:
            cmd.inject_subparser(subparser)


class CliSubcommandSetOperation(SubparserInjectable):
    command_id: str = "set-operation"

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description=("Applies a set operation to the two prompts."),
            help=("Applies a set operation to the two prompts."),
        )

        parser.add_argument(
            "-a",
            "--fixed-prompt",
            type=argparse.FileType("r"),
            help=("Feed the elements of set a from a file or stdin."),
        )

        parser.add_argument(
            "-i",
            "--interactive",
            default=False,
            action="store_true",
            help=(
                "Provides the REPL interface to play with prompts. "
                "The program behaves like the Python interpreter."
            ),
        )

        subparser = parser.add_subparsers(
            title=cls.command_id,
            description="Applies a set operation to the two prompts.",
            help="List of available set operations to the two prompts.",
            dest="set_op_id",
            metavar="SET_OPERATION",
            required=True,
        )

        subparser.add_parser(
            SetCalculatorWrapper.union,
            help=("Combines all tokens of two prompts into one."),
        )

        subparser.add_parser(
            SetCalculatorWrapper.intersection,
            help=("Extracts only the tokens that are common to two prompts."),
        )

        subparser.add_parser(
            SetCalculatorWrapper.difference,
            help=("Excludes the tokens of the second prompt from the first one."),
        )

        subparser.add_parser(
            SetCalculatorWrapper.symmetric_difference,
            help=("Collects only tokens that are in only one of the two prompts."),
        )


class CliSubcommandParserV2(SubparserInjectable):
    command_id: str = "parserv2"

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id,
            formatter_class=SaniproHelpFormatter,
            help=("Switch to use another version of the parser instead."),
            description=(
                "Switch to use another version of the parser instead. "
                "It only parses the prompt and does nothing at all."
            ),
        )

        parser.add_argument(
            "-i",
            "--interactive",
            default=False,
            action="store_true",
            help=(
                "Provides the REPL interface to play with prompts. "
                "The program behaves like the Python interpreter."
            ),
        )


class CliArgsNamespaceDemo(CliArgsNamespaceDefault):
    """Custom subcommand implementation by user"""

    # global options
    interactive: bool
    exclude: Sequence[str]
    roundup = 2
    replace_to: str
    mask: Sequence[str]

    # 'dest' name for general operations
    operation_id = str  # may be 'filter', 'set_op', and more

    # list of 'dest' name for subcommands
    filter_id: str | None = None  # may be 'unique', 'reset' and more
    set_op_id: str | None = None  # may be 'union', 'xor', and more

    # subcommands.filter options
    reverse: bool
    seed: int
    value: float
    similar_method: str
    sort_all_method: str
    kruskal: bool
    prim: bool

    # for tfind subcommand
    infile: typing.TextIO
    key_field: int
    value_field: int
    field_delimiter: str
    tempdir: str
    clipboard: bool
    fixed_prompt: typing.TextIO

    def is_parser_v2(self) -> bool:
        return self.operation_id == CliSubcommandParserV2.command_id

    def is_filter(self) -> bool:
        return self.operation_id == CliSubcommandFilter.command_id

    def is_tfind(self) -> bool:
        return self.operation_id == CliSubcommandSearchTag.command_id

    def is_set_operation(self) -> bool:
        return self.operation_id == CliSubcommandSetOperation.command_id

    @classmethod
    def _do_append_parser(cls, parser: SaniproArgumentParser) -> None:
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

    @classmethod
    def _do_append_subparser(cls, parser: SaniproArgumentParser) -> None:
        subparser = parser.add_subparsers(
            title="operations", dest="operation_id", required=True
        )

        classes: list[type[SubparserInjectable]] = [
            CliSubcommandFilter,
            CliSubcommandSetOperation,
            CliSubcommandParserV2,
            CliSubcommandSearchTag,
        ]

        for cmd in classes:
            cmd.inject_subparser(subparser)


def prepare_readline() -> None:
    """Prepare readline for the interactive mode."""
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


class ShowCliStatMixin(StatShowable):
    def _show_cli_stat(
        self,
        detector: type[PromptDifferenceDetector],
        before: MutablePrompt,
        after: MutablePrompt,
    ) -> None:
        for line in detector(before, after).get_summary():
            logger.info("(statistics) %s", line)


class RunnerTagFindNonInteractive(ExecuteSingle, RunnerNonInteractive):
    """Represents the runner specialized for the filtering mode."""

    def __init__(
        self,
        pipeline: PromptPipeline,
        tags_n_count: dict[str, str],
        strategy: InputStrategy,
    ) -> None:
        self._pipeline = pipeline
        self._input_strategy = strategy
        self.tags_n_count: dict[str, str] = tags_n_count
        self._histfile = ""

    @classmethod
    def create_from_csv(
        cls,
        pipeline: PromptPipeline,
        text: typing.TextIO,
        strategy: InputStrategy,
        delim: str,
        key_idx: int,
        value_idx: int,
    ) -> Self:
        """Import the key-value storage from a comma-separated file.
        The index starts from 1. This is because common traditional command-line
        utilities assume the field index originates from 1."""

        if key_idx == value_idx:
            raise ValueError("impossible to specify the same field number")
        if key_idx < 1 or value_idx < 1:
            raise ValueError("field number must be 1 or more")
        key_idx -= 1
        value_idx -= 1

        dict_: dict[str, str] = {}
        with text as fp:
            try:
                for row in map(
                    lambda line: line.split(delim),
                    map(lambda ln: ln.strip("\n").replace("_", " "), fp.readlines()),
                ):
                    dict_ |= {row[key_idx]: row[value_idx]}
            except IndexError:
                raise IndexError("failed to get the element of the row number")

        return cls(pipeline, dict_, strategy)

    def _execute_single_inner(self, source: str) -> str:
        tokens = [
            "%s\t%s" % (str(token), self.tags_n_count.get(token.name, "null"))
            for token in self._pipeline.tokenize(source)
        ]
        return self._pipeline.delimiter.sep_output.join(tokens)


class RunnerTagFindInteractive(ExecuteSingle, RunnerInteractive):
    """Represents the runner specialized for the filtering mode."""

    def __init__(
        self,
        pipeline: PromptPipeline,
        tags_n_count: dict[str, str],
        strategy: InputStrategy,
        tempdir: str,
        use_clipboard: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._input_strategy = strategy
        self.tags_n_count: dict[str, str] = tags_n_count
        self.tempdir = tempdir
        self._histfile = ""
        self._use_clipboard = use_clipboard

    @classmethod
    def create_from_csv(
        cls,
        pipeline: PromptPipeline,
        text: typing.TextIO,
        strategy: InputStrategy,
        delim: str,
        key_idx: int,
        value_idx: int,
        tempdir: str,
        use_clipboard: bool,
    ) -> Self:
        """Import the key-value storage from a comma-separated file.
        The index starts from 1. This is because common traditional command-line
        utilities assume the field index originates from 1."""

        if key_idx == value_idx:
            raise ValueError("impossible to specify the same field number")
        if key_idx < 1 or value_idx < 1:
            raise ValueError("field number must be 1 or more")
        key_idx -= 1
        value_idx -= 1

        dict_: dict[str, str] = {}

        def _replace_underscore(line: str) -> str:
            return line.strip("\n").replace("_", " ")

        with text as fp:
            try:
                for row in map(
                    lambda line: line.split(delim),
                    map(_replace_underscore, fp.readlines()),
                ):
                    dict_ |= {row[key_idx]: row[value_idx]}
            except IndexError:
                raise IndexError("failed to get the element of the row number")

        return cls(pipeline, dict_, strategy, tempdir, use_clipboard)

    def _on_init(self) -> None:
        histfile = None
        with tempfile.NamedTemporaryFile(delete=False, dir=self.tempdir) as fp:
            histfile = fp.name
            for key in self.tags_n_count.keys():
                fp.write(f"{key}\n".encode())

        # so that the file is deleted after the program exits
        self._histfile = histfile

        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            logger.exception(f"failed to read history file: {histfile}")

    def _on_exit(self) -> None:
        if self._histfile:
            os.remove(self._histfile)

    def _copy_to_clipboard(self, text: str) -> None:
        """Check if clipboard API is available and copy to clipboard, if possible."""
        if os.name == "nt":
            subprocess.run(["clip.exe"], input=text.encode())
        elif os.name == "posix":
            if "Microsoft" in open("/proc/version").read():
                subprocess.run(["clip.exe"], input=text.encode())
            else:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"], input=text.encode()
                )

    def _execute_single_inner(self, source: str) -> str:
        tokens = [
            "%s\t%s" % (str(token), self.tags_n_count.get(token.name, "null"))
            for token in self._pipeline.tokenize(source)
        ]
        selialized = self._pipeline.delimiter.sep_output.join(tokens)

        if self._use_clipboard:
            try:
                self._copy_to_clipboard(selialized)
            except subprocess.SubprocessError:
                logger.error("failed to copy to clipboard")
            except FileNotFoundError:
                logger.debug("clipboard API is not available")

        return selialized


class RunnerFilterInteractive(ExecuteSingle, ShowCliStatMixin, RunnerInteractive):
    """Represents the runner specialized for the filtering mode."""

    def __init__(
        self,
        pipeline: PromptPipeline,
        strategy: InputStrategy,
        detector: type[PromptDifferenceDetector],
    ) -> None:
        self._pipeline = pipeline
        self._token_cls = pipeline.token_cls
        self._input_strategy = strategy

        self._detector_cls = detector

    def _execute_single_inner(self, source: str) -> str:
        unparsed = self._pipeline.tokenize(source)
        parsed = self._pipeline.execute(source)

        self._show_cli_stat(self._detector_cls, unparsed, parsed)

        selialized = str(self._pipeline)
        return selialized


class RunnerFilterNonInteractive(ExecuteSingle, RunnerNonInteractive):
    """Represents the runner specialized for the filtering mode."""

    def __init__(self, pipeline: PromptPipeline, strategy: InputStrategy) -> None:
        self._pipeline = pipeline
        self._token_cls = pipeline.token_cls
        self._input_strategy = strategy

    def _execute_single_inner(self, source: str) -> str:
        self._pipeline.execute(source)
        return str(self._pipeline)


class RunnerSetOperationInteractiveDual(
    ExecuteDual, ShowCliStatMixin, RunnerInteractive
):
    """Represents the runner specialized for the set operation.

    In set operation mode, the total number of tokens will be more
    than prompt A or prompt B. Thus it is reasonable that showing
    the difference between both prompt A and result, and prompt B and result."""

    def __init__(
        self,
        pipeline: PromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
        detector: type[PromptDifferenceDetector],
    ) -> None:
        self._pipeline = pipeline
        self._token_cls = pipeline.token_cls
        self._input_strategy = strategy

        self._calculator = calculator
        self._detector_cls = detector

    def _execute_multi_inner(self, first: str, second: str) -> str:
        prompt_first_before = self._pipeline.tokenize(first)
        prompt_second_before = self._pipeline.tokenize(second)

        prompt = [
            self._token_cls(name=x.name, weight=x.weight)
            for x in self._calculator.do_math(prompt_first_before, prompt_second_before)
        ]

        logger.info("(statistics) prompt A <> result")
        self._show_cli_stat(self._detector_cls, prompt_first_before, prompt)
        logger.info("(statistics) prompt B <> result")
        self._show_cli_stat(self._detector_cls, prompt_second_before, prompt)

        selialized = self._pipeline.delimiter.sep_output.join(
            str(token) for token in prompt
        )
        return selialized


class RunnerSetOperationIteractiveMono(
    ExecuteSingle, ShowCliStatMixin, RunnerNonInteractive
):

    def __init__(
        self,
        pipeline: PromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
        detector: type[PromptDifferenceDetector],
        fixed_prompt: MutablePrompt,
    ) -> None:
        self._pipeline = pipeline
        self._token_cls = pipeline.token_cls
        self._input_strategy = strategy

        self._calculator = calculator
        self._detector_cls = detector
        self._fixed_prompt = fixed_prompt

    @classmethod
    def create_from_text(
        cls,
        pipeline: PromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
        detector: type[PromptDifferenceDetector],
        text: typing.TextIO,
    ) -> Self:
        fixed_prompt = pipeline.tokenize(text.read())
        return cls(pipeline, strategy, calculator, detector, fixed_prompt)

    def _execute_single_inner(self, source: str) -> str:
        prompt_second_before = self._pipeline.tokenize(source)

        prompt = [
            self._token_cls(name=x.name, weight=x.weight)
            for x in self._calculator.do_math(self._fixed_prompt, prompt_second_before)
        ]

        selialized = self._pipeline.delimiter.sep_output.join(
            str(token) for token in prompt
        )
        return selialized


class CliCommandsDemo(CliCommands):
    def __init__(self, args: CliArgsNamespaceDemo) -> None:
        self._args = args

    def _get_input_strategy(self) -> InputStrategy:
        if not self._args.interactive:
            return inputs.DirectInputStrategy()

        ps1 = self._args.ps1
        ps2 = self._args.ps2

        return (
            inputs.OnelineInputStrategy(ps1)
            if self._args.one_line
            else inputs.MultipleInputStrategy(ps1, ps2)
        )

    def _initialize_parser(self) -> type[ParserInterface]:
        if self._args.is_tfind():
            return DummyParser
        else:
            return ParserV2 if self._args.is_parser_v2() else ParserV1

    def _initialize_tokenizer(self) -> PromptTokenizer:
        token_cls = TokenInteractive if self._args.interactive else TokenNonInteractive
        parser_cls = self._initialize_parser()

        tokenizer_cls = None
        if self._args.is_parser_v2():
            tokenizer_cls = PromptTokenizerV2
        elif self._args.is_tfind():
            tokenizer_cls = PromptTokenizerV2
        else:
            tokenizer_cls = PromptTokenizerV1
        delimiter = Delimiter(self._args.input_delimiter, self._args.output_delimiter)
        return tokenizer_cls(parser_cls, token_cls, delimiter)

    def _initialize_filter_pipeline(self) -> FilterExecutor:
        filterpipe = FilterExecutor()
        filterpipe.append_command(CliRoundUpCommand(self._args.roundup))

        if self._args.filter_id is not None:
            command_map = self._command_map()
            filterpipe.append_command(command_map[self._args.filter_id]())

        if self._args.exclude:
            filterpipe.append_command(CliExcludeCommand(self._args.exclude))

        return filterpipe

    def _initialize_pipeline(self) -> type[PromptPipeline]:
        return PromptPipelineV2 if self._args.is_parser_v2() else PromptPipelineV1

    def _initialize_runner(self, pipe: PromptPipeline) -> CliRunnable:
        """Returns a runner."""
        input_strategy = self._get_input_strategy()

        if not self._args.is_tfind():
            cli_hooks.on_init.append(prepare_readline)
            cli_hooks.execute(cli_hooks.on_init)

        if self._args.is_filter() or self._args.is_parser_v2():
            if self._args.interactive:
                return RunnerFilterInteractive(
                    pipe, input_strategy, PromptDifferenceDetector
                )
            return RunnerFilterNonInteractive(pipe, input_strategy)
        elif self._args.is_set_operation():
            calculator = SetCalculatorWrapper.create_from(self._args.set_op_id)
            if self._args.interactive:
                return RunnerSetOperationInteractiveDual(
                    pipe, input_strategy, calculator, PromptDifferenceDetector
                )
            else:
                return RunnerSetOperationIteractiveMono.create_from_text(
                    pipe,
                    input_strategy,
                    calculator,
                    PromptDifferenceDetector,
                    self._args.fixed_prompt,
                )
        elif self._args.is_tfind():
            if self._args.interactive:
                return RunnerTagFindInteractive.create_from_csv(
                    pipeline=pipe,
                    text=self._args.infile,
                    strategy=input_strategy,
                    delim=self._args.field_delimiter,
                    key_idx=self._args.key_field,
                    value_idx=self._args.value_field,
                    tempdir=self._args.tempdir,
                    use_clipboard=self._args.clipboard,
                )
            return RunnerTagFindNonInteractive.create_from_csv(
                pipeline=pipe,
                text=self._args.infile,
                strategy=input_strategy,
                delim=self._args.field_delimiter,
                key_idx=self._args.key_field,
                value_idx=self._args.value_field,
            )
        else:  # default
            raise NotImplementedError

    def _get_pipeline(self) -> PromptPipeline:
        """This is a pipeline for the purpose of showcasing.
        Since all the parameters of each command is variable, the command
        sacrifices the composability.
        It is good for you to create your own pipeline, and name it
        so you can use it as a preset."""

        tokenizer = self._initialize_tokenizer()
        filterpipe = self._initialize_filter_pipeline()
        return self._initialize_pipeline()(tokenizer, filterpipe)

    def _get_runner(self) -> CliRunnable:
        pipe = self._get_pipeline()
        runner = self._initialize_runner(pipe)
        return runner

    def to_runner(self) -> CliRunnable:
        return self._get_runner()

    def _command_map(self) -> dict[str, Callable]:
        args = self._args

        command_ids = [cmd.command_id for cmd in CliSubcommandFilter.filter_classes]
        command_funcs = (
            lambda: CliMaskCommand(args.mask, args.replace_to),
            lambda: CliRandomCommand(args.seed),
            lambda: CliResetCommand(args.value),
            lambda: CliSimilarCommand.create_from_cmd(cmd=args, reverse=args.reverse),
            lambda: CliSortAllCommand.create_from_cmd(cmd=args, reverse=args.reverse),
            lambda: CliSortCommand(args.reverse),
            lambda: CliUniqueCommand(args.reverse),
        )
        command_map = dict(zip(command_ids, command_funcs, strict=True))

        return command_map


def app():
    try:
        args = CliArgsNamespaceDemo.from_sys_argv(sys.argv[1:])
        cli_commands = CliCommandsDemo(args)

        log_level = cli_commands.get_logger_level()
        logger_root.setLevel(log_level)

        for key, val in args.__dict__.items():
            logger.debug(f"(settings) {key} = {val!r}")

        runner = cli_commands.to_runner()
        runner.run()
    except Exception as e:
        logger.exception(f"error: {e}")
    finally:
        sys.exit(1)


if __name__ == "__main__":
    app()
