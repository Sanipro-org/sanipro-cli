import argparse
import atexit
import logging
import os
import readline
import sys
import typing
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

from sanipro.abc import IPipelineResult, IPromptPipeline, MutablePrompt
from sanipro.compatible import Self
from sanipro.token_types import SupportedInTokenType, SupportedOutTokenType

if TYPE_CHECKING:
    from sanipro.converter_context import TokenMap

from sanipro.delimiter import Delimiter
from sanipro.diff import PromptDifferenceDetector
from sanipro.filter_exec import FilterExecutor
from sanipro.filters.abc import ReordererStrategy
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
from sanipro.filters.translate import TranslateTokenTypeCommand
from sanipro.filters.unique import UniqueCommand
from sanipro.filters.utils import (
    sort_by_length,
    sort_by_ord_sum,
    sort_by_weight,
    sort_lexicographically,
)
from sanipro.logger import logger, logger_root
from sanipro.promptset import SetCalculatorWrapper

from saniprocli import cli_hooks, inputs
from saniprocli.abc import CliRunnable, InputStrategy, RunnerFilter, RunnerSetOperation
from saniprocli.cli_runner import (
    ExecuteMultiple,
    ExecuteSingle,
    RunnerDeclarative,
    RunnerInteractive,
)
from saniprocli.color import style
from saniprocli.commands import (
    CliArgsNamespaceDefault,
    CliCommand,
    CliCommands,
    SubparserInjectable,
)
from saniprocli.help_formatter import SaniproHelpFormatter
from saniprocli.sanipro_argparse import SaniproArgumentParser
from saniprocli.textutils import ClipboardHandler

logging.basicConfig(
    format=style(
        ("[%(levelname)s] %(module)s/%(funcName)s (%(lineno)d):") + " %(message)s"
    ),
    datefmt=r"%Y-%m-%d %H:%M:%S",
)


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
        try:
            return self.commands.list_commands()[method]
        except KeyError:
            raise ModuleNotFoundError


class CliExcludeCommand(CliCommand):
    command_id: str = "exclude"

    def __init__(self, excludes: Sequence[str]):
        self.command = ExcludeCommand(excludes)


class SimilarModuleMapper(ModuleMapper):
    NAIVE = CmdModuleTuple("naive", NaiveReorderer)
    GREEDY = CmdModuleTuple("greedy", GreedyReorderer)
    KRUSKAL = CmdModuleTuple("kruskal", KruskalMSTReorderer)
    PRIM = CmdModuleTuple("prim", PrimMSTReorderer)


class CliSimilarCommand(CliCommand):
    command_id: str = "similar"

    def __init__(self, reorderer: ReordererStrategy, *, reverse=False):
        self.command = SimilarCommand(reorderer, reverse=reverse)

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
            required=True,
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
        cls, method: str = SimilarModuleMapper.GREEDY.key
    ) -> type[ReordererStrategy]:
        mapper = ModuleMatcher(SimilarModuleMapper)
        return mapper.match(method)

    @classmethod
    def get_class(cls, cmd: "CliArgsNamespaceDemo") -> type[ReordererStrategy]:
        """Matches the methods specified on the command line
        to the names of concrete classes.
        Searches other than what the strategy uses MST."""

        query = cmd.similar_method
        if query == "mst":
            if cmd.kruskal:
                query = "kruskal"
            elif cmd.prim:
                query = "prim"

        return cls._query_strategy(method=query)

    @classmethod
    def get_reorderer(cls, cmd: "CliArgsNamespaceDemo") -> ReordererStrategy:
        """Instanciate one reorder function from the parsed result."""

        selected_cls = cls.get_class(cmd)
        return selected_cls(strategy=SequenceMatcherSimilarity())

    @classmethod
    def create_from_cmd(cls, cmd: "CliArgsNamespaceDemo") -> Self:
        """Alternative method."""

        return cls(reorderer=cls.get_reorderer(cmd), reverse=cmd.reverse)


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

    def __init__(self, key: Callable, reverse: bool = False):
        self.command = SortAllCommand(key, reverse)

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
            required=True,
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
    def _query_strategy(
        cls, method: str = SortAllModuleMapper.LEXICOGRAPHICAL.key
    ) -> Callable:
        """Matches `method` to the name of a concrete class."""
        mapper = ModuleMatcher(SortAllModuleMapper)
        try:
            return mapper.match(method)
        except KeyError:
            raise ValueError("method name is not found.")

    @classmethod
    def create_from_cmd(cls, cmd: "CliArgsNamespaceDemo") -> Self:
        """Alternative method."""
        method = cmd.sort_all_method
        partial = cls._query_strategy(method)
        return cls(partial, reverse=cmd.reverse)


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

        parser.add_argument(
            "-c",
            "--clipboard",
            default=False,
            action="store_true",
            help="Copy the result to the clipboard if possible.",
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

        parser.add_argument(
            "-c",
            "--clipboard",
            default=False,
            action="store_true",
            help="Copy the result to the clipboard if possible.",
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


class CliArgsNamespaceDemo(CliArgsNamespaceDefault):
    """Custom subcommand implementation by user"""

    # global options
    input_type: str
    output_type: str
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

    clipboard: bool
    fixed_prompt: typing.TextIO
    config: str

    def is_filter(self) -> bool:
        return self.operation_id == CliSubcommandFilter.command_id

    def is_set_operation(self) -> bool:
        return self.operation_id == CliSubcommandSetOperation.command_id

    @classmethod
    def _do_append_parser(cls, parser: SaniproArgumentParser) -> None:

        parser.add_argument(
            "-d",
            "--input-type",
            type=str,
            choices=SupportedInTokenType.choises(),
            default="a1111compat",
            help=("Preferred token type for the original prompts."),
        )

        parser.add_argument(
            "-s",
            "--output-type",
            type=str,
            choices=SupportedOutTokenType.choises(),
            default="a1111compat",
            help=("Preferred token type for the processed prompts."),
        )

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
            "-c",
            "--config",
            type=str,
            default=None,
            help=("Specifies a config file for each token type."),
        )

    @classmethod
    def _do_append_subparser(cls, parser: SaniproArgumentParser) -> None:
        subparser = parser.add_subparsers(
            title="operations", dest="operation_id", required=True
        )

        classes: list[type[SubparserInjectable]] = [
            CliSubcommandFilter,
            CliSubcommandSetOperation,
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


class StatisticsHandler:
    @staticmethod
    def show_cli_stat(result: IPipelineResult) -> None:
        """Explains what has changed in the unprocessed/processsed prompts."""

        for line in result.get_summary():
            logger.info("(statistics) %s", line)


class RunnerFilterInteractive(ExecuteSingle, RunnerInteractive, RunnerFilter):
    """Represents the runner specialized for the filtering mode."""

    def __init__(
        self,
        pipeline: IPromptPipeline,
        strategy: InputStrategy,
        detector: type[PromptDifferenceDetector],
        use_clipboard: bool,
    ) -> None:
        super().__init__(pipeline, strategy)
        self._token_cls = pipeline.tokenizer.token_cls

        self._detector_cls = detector
        self._use_clipboard = use_clipboard

    def _execute_single_inner(self, source: str) -> str:
        result = self._pipeline.execute(source)
        StatisticsHandler.show_cli_stat(result)

        selialized = str(self._pipeline)

        if self._use_clipboard:
            ClipboardHandler.copy_to_clipboard(selialized)

        return selialized


class RunnerFilterDeclarative(ExecuteSingle, RunnerDeclarative, RunnerFilter):
    """Represents the runner specialized for the filtering mode."""

    def __init__(self, pipeline: IPromptPipeline, strategy: InputStrategy) -> None:
        super().__init__(pipeline, strategy)
        self._token_cls = pipeline.tokenizer.token_cls

    def _execute_single_inner(self, source: str) -> str:
        self._pipeline.execute(source)
        return str(self._pipeline)


class RunnerSetOperationInteractiveDual(
    ExecuteMultiple, RunnerInteractive, RunnerSetOperation
):
    def __init__(
        self,
        pipeline: IPromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
        detector: type[PromptDifferenceDetector],
        use_clipboard: bool,
    ) -> None:
        super().__init__(pipeline, strategy, calculator)

        self._tokenizer = pipeline.tokenizer
        self._detector_cls = detector
        self._use_clipboard = use_clipboard

    def _execute_multi_inner(self, first: str, second: str) -> str:
        from sanipro.pipelineresult import PipelineResult

        prompt_first_before = self._tokenizer.tokenize_prompt(first)
        prompt_second_before = self._tokenizer.tokenize_prompt(second)

        prompt = [
            self._tokenizer.token_cls(name=x.name, weight=x.weight)
            for x in self._calculator.do_math(prompt_first_before, prompt_second_before)
        ]

        logger.info("(statistics) prompt A <> result")
        StatisticsHandler.show_cli_stat(PipelineResult(prompt_first_before, prompt))

        logger.info("(statistics) prompt B <> result")
        StatisticsHandler.show_cli_stat(PipelineResult(prompt_second_before, prompt))

        selialized = str(self._pipeline.new(prompt))

        if self._use_clipboard:
            ClipboardHandler.copy_to_clipboard(selialized)

        return selialized


class RunnerSetOperationDeclarativeMono(
    ExecuteSingle, RunnerDeclarative, RunnerSetOperation
):
    def __init__(
        self,
        pipeline: IPromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
        detector: type[PromptDifferenceDetector],
        fixed_prompt: MutablePrompt,
        use_clipboard: bool,
    ) -> None:
        super().__init__(pipeline, strategy, calculator)
        self._tokenizer = self._pipeline.tokenizer

        self._detector_cls = detector
        self._fixed_prompt = fixed_prompt
        self._use_clipboard = use_clipboard

    @classmethod
    def create_from_text(
        cls,
        pipeline: IPromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
        detector: type[PromptDifferenceDetector],
        text: typing.TextIO,
        use_clipboard: bool,
    ) -> Self:
        fixed_prompt = pipeline.tokenizer.tokenize_prompt(text.read())
        return cls(
            pipeline, strategy, calculator, detector, fixed_prompt, use_clipboard
        )

    def _execute_single_inner(self, source: str) -> str:
        prompt_second_before = self._tokenizer.tokenize_prompt(source)

        prompt = [
            self._tokenizer.token_cls(name=x.name, weight=x.weight)
            for x in self._calculator.do_math(self._fixed_prompt, prompt_second_before)
        ]

        selialized = str(self._pipeline.new(prompt))

        if self._use_clipboard:
            ClipboardHandler.copy_to_clipboard(selialized)

        return selialized


class CliCommandsDemo(CliCommands):
    def __init__(self, args: CliArgsNamespaceDemo) -> None:
        from sanipro.converter_context import get_config

        self._args = args
        self._config = get_config(self._args.config)
        self.input_type = self._config.get_input_token_class(self._args.input_type)
        self.output_type = self._config.get_output_token_class(self._args.output_type)

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

    def _initialize_formatter(self, token_map: "TokenMap") -> Callable:
        """Initialize formatter function which takes an only 'Token' class.
        Note when 'csv' is chosen as Token, the token_map.formatter is
        a partial function."""

        formatter = token_map.formatter

        if token_map.type_name == "csv":
            new_formatter = formatter(token_map.field_separator)
            formatter = new_formatter

        return formatter

    def _initialize_delimiter(self, itype: "TokenMap") -> Delimiter:
        its = self._config.get_input_token_separator(self._args.input_type)
        ots = self._config.get_output_token_separator(self._args.output_type)
        ifs = itype.field_separator

        delimiter = Delimiter(its, ots, ifs)
        return delimiter

    def _initialize_pipeline(self) -> IPromptPipeline:
        from sanipro.pipeline_v1 import PromptPipelineV1

        formatter = self._initialize_formatter(self.output_type)
        filter_pipe = self._initialize_filter_pipeline()
        delimiter = self._initialize_delimiter(self.input_type)

        parser = self.input_type.parser(delimiter)
        token_type = self.output_type.token_type
        tokenizer = self.input_type.tokenizer(parser, token_type)
        return PromptPipelineV1(tokenizer, filter_pipe, formatter)

    def _initialize_filter_pipeline(self) -> FilterExecutor:
        filterpipe = FilterExecutor()
        filterpipe.append_command(CliRoundUpCommand(self._args.roundup).command)

        if self._args.filter_id is not None:
            command_map = self._command_map()
            filterpipe.append_command(command_map[self._args.filter_id]().command)

        if self._args.exclude:
            filterpipe.append_command(CliExcludeCommand(self._args.exclude).command)

        # add filter to for converting token type
        filterpipe.append_command(
            TranslateTokenTypeCommand(self.output_type.token_type)
        )

        return filterpipe

    def _initialize_runner(self, pipe: IPromptPipeline) -> CliRunnable:
        """Returns a runner."""
        input_strategy = self._get_input_strategy()

        cli_hooks.on_init.append(prepare_readline)
        cli_hooks.execute(cli_hooks.on_init)

        if self._args.is_filter():
            if self._args.interactive:
                return RunnerFilterInteractive(
                    pipe, input_strategy, PromptDifferenceDetector, self._args.clipboard
                )
            return RunnerFilterDeclarative(pipe, input_strategy)

        elif self._args.is_set_operation():
            calculator = SetCalculatorWrapper.create_from(self._args.set_op_id)
            if self._args.interactive:
                return RunnerSetOperationInteractiveDual(
                    pipe,
                    input_strategy,
                    calculator,
                    PromptDifferenceDetector,
                    self._args.clipboard,
                )
            else:
                return RunnerSetOperationDeclarativeMono.create_from_text(
                    pipe,
                    input_strategy,
                    calculator,
                    PromptDifferenceDetector,
                    self._args.fixed_prompt,
                    self._args.clipboard,
                )
        else:  # default
            raise NotImplementedError

    def _get_pipeline(self) -> IPromptPipeline:
        """This is a pipeline for the purpose of showcasing.
        Since all the parameters of each command is variable, the command
        sacrifices the composability.
        It is good for you to create your own pipeline, and name it
        so you can use it as a preset."""

        pipeline_cls = self._initialize_pipeline()
        return pipeline_cls

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
            lambda: CliSimilarCommand.create_from_cmd(args),
            lambda: CliSortAllCommand.create_from_cmd(args),
            lambda: CliSortCommand(args.reverse),
            lambda: CliUniqueCommand(args.reverse),
        )
        command_map = dict(zip(command_ids, command_funcs, strict=True))

        return command_map


def app():
    args = CliArgsNamespaceDemo.from_sys_argv(sys.argv[1:])
    cli_commands = CliCommandsDemo(args)

    log_level = cli_commands.get_logger_level()
    logger_root.setLevel(log_level)

    try:
        runner = cli_commands.to_runner()
    except AttributeError as e:
        logger.error(f"error: {e}")
        exit(1)

    runner.run()


if __name__ == "__main__":
    app()
