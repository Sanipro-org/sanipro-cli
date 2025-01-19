"""
command          := prog_name global_options? filter_spec+
prog_name        := "sanipro"
global_options   := "-" { "x" | ("v")+ | "u" | "s" | "p" | "l" | "i" | "d" | "c" }
filter_spec      := filter_name filter_args
filter_name      := "mask"
                  | "random"
                  | "reset"
                  | "similar"
                  | "sort-all"
                  | "sort"
                  | "unique"
"""

import atexit
import logging
import os
import readline
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from sanipro.abc import IPipelineResult, IPromptPipeline
from sanipro.compatible import Self
from sanipro.token_types import SupportedInTokenType, SupportedOutTokenType

if TYPE_CHECKING:
    from sanipro.converter_context import TokenMap

import abc
from collections.abc import Callable

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

from saniprocli import cli_hooks, inputs
from saniprocli.abc import CliRunnable, InputStrategy, RunnerFilter
from saniprocli.cli_runner import ExecuteSingle, RunnerDeclarative, RunnerInteractive
from saniprocli.color import style
from saniprocli.commands import CliCommands
from saniprocli.help_formatter import SaniproHelpFormatter
from saniprocli.sanipro_argparse import SaniproArgumentParser
from saniprocli.textutils import ClipboardHandler

logging.basicConfig(
    format=style(
        ("[%(levelname)s] %(module)s/%(funcName)s (%(lineno)d):") + " %(message)s"
    ),
    datefmt=r"%Y-%m-%d %H:%M:%S",
)


class SubparserInjectable(abc.ABC):
    """The trait with the ability to inject a subparser."""

    @classmethod
    @abc.abstractmethod
    def get_parser(cls) -> SaniproArgumentParser:
        """Injects subparser."""


class CliCommand(SubparserInjectable):
    """The wrapper class for the filter commands
    with the addition of subparser."""

    command_id: str

    @classmethod
    @abc.abstractmethod
    def get_parser(cls) -> SaniproArgumentParser:
        """Does nothing by default."""


class CliExcludeCommand(CliCommand):
    command_id: str = "exclude"
    exclude: Sequence[str]

    def __init__(self, excludes: Sequence[str]):
        self.command = ExcludeCommand(excludes)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(cls.command_id)
        subcommand.add_argument(
            "-x",
            "--exclude",
            type=str,
            nargs="*",
            help="Exclude this token from the original prompt. Multiple options can be specified.",
        )
        return subcommand


class CliSimilarCommand(CliCommand):
    command_id: str = "similar"
    method: ReordererStrategy
    reverse: bool

    def __init__(self, reorderer: ReordererStrategy, *, reverse=False):
        self.command = SimilarCommand(reorderer, reverse=reverse)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        parser = SaniproArgumentParser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description="Reorders tokens with their similarity.",
        )
        parser.add_argument(
            "-r",
            "--reverse",
            default=False,
            action="store_true",
            help="With reversed order.",
        )

        def _matcher(method: str) -> ReordererStrategy:
            """Instanciate one reorder function from the parsed result."""

            def _inner() -> type[ReordererStrategy]:
                """Matches the methods specified on the command line
                to the names of concrete classes.
                Searches other than what the strategy uses MST."""

                if method == "naive":
                    return NaiveReorderer
                elif method == "greedy":
                    return GreedyReorderer
                elif method == "kruskal":
                    return KruskalMSTReorderer
                return PrimMSTReorderer

            Reorderer = _inner()
            return Reorderer(strategy=SequenceMatcherSimilarity())

        parser.add_argument(
            "-m",
            "--method",
            choices=["naive", "greedy", "kruskal", "prim"],
            default="prim",
            type=_matcher,
            help="The available method to sort the tokens.",
        )

        return parser


class CliMaskCommand(CliCommand):
    command_id: str = "mask"
    mask: Sequence[str]
    replace_to: str

    def __init__(self, excludes: Sequence[str], replace_to: str):
        self.command = MaskCommand(excludes, replace_to)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        parser = SaniproArgumentParser(
            cls.command_id,
            description="Mask words specified with another word (optional).",
            formatter_class=SaniproHelpFormatter,
            epilog="Note that you can still use the global `--exclude` option"
            "as well as this filter.",
        )
        parser.add_argument("mask", nargs="*", type=str, help="Masks this word.")
        parser.add_argument(
            "-t",
            "--replace-to",
            type=str,
            default=r"%%%",
            help="The new character or string replaced to.",
        )
        return parser


class CliRandomCommand(CliCommand):
    command_id: str = "random"
    seed: int

    def __init__(self, seed: int | None = None):
        self.command = RandomCommand(seed)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description="Shuffles all the prompts altogether.",
        )
        subcommand.add_argument(
            "-b",
            "--seed",
            default=None,
            type=int,
            help="Fixed randomness to this value.",
        )
        return subcommand


class CliResetCommand(CliCommand):
    command_id: str = "reset"
    value: float

    def __init__(self, new_value: float | None = None) -> None:
        self.command = ResetCommand(new_value)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description="Initializes all the weight of the tokens.",
        )
        subcommand.add_argument(
            "-v",
            "--value",
            default=1.0,
            type=float,
            help="Fixes the weight to this value.",
        )
        return subcommand


class CliRoundUpCommand(CliCommand):
    command_id: str = "roundup"

    def __init__(self, digits: int):
        self.command = RoundUpCommand(digits)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            cls.command_id, formatter_class=SaniproHelpFormatter
        )
        subcommand.add_argument(
            "-u",
            "--roundup",
            type=int,
            default=2,
            help="All the token with weights (x > 1.0 or x < 1.0) will be rounded up to n digit(s).",
        )
        return subcommand


class CliSortCommand(CliCommand):
    command_id: str = "sort"
    reverse: bool

    def __init__(self, reverse: bool = False):
        self.command = SortCommand(reverse)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description="Reorders duplicate tokens.",
            epilog="This command reorders tokens with their weights by default.",
        )
        subcommand.add_argument(
            "-r", "--reverse", action="store_true", help="With reversed order."
        )
        return subcommand


class CliSortAllCommand(CliCommand):
    command_id: str = "sort-all"
    reverse: bool
    method: Callable

    def __init__(self, key: Callable, reverse: bool = False):
        self.command = SortAllCommand(key, reverse)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        def _matcher(method: str) -> Callable:
            """Matches `method` to the name of a concrete class."""
            if method == "length":
                return sort_by_length
            elif method == "weight":
                return sort_by_weight
            elif method == "ord-sum":
                return sort_by_ord_sum
            return sort_lexicographically

        parser = SaniproArgumentParser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description="Reorders all the prompts.",
        )
        parser.add_argument(
            "-r", "--reverse", action="store_true", help="With reversed order."
        )
        parser.add_argument(
            "-m",
            "--method",
            choices=["lexicographical", "length", "weight", "ord-sum"],
            default="lexicographical",
            type=_matcher,
            help="The available method to sort the tokens.",
        )

        return parser


class CliUniqueCommand(CliCommand):
    command_id: str = "unique"
    reverse: bool

    def __init__(self, reverse: bool):
        self.command = UniqueCommand(reverse)

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        parser = SaniproArgumentParser(
            cls.command_id,
            formatter_class=SaniproHelpFormatter,
            description="Removes duplicated tokens, and uniquify them.",
        )
        parser.add_argument(
            "-r",
            "--reverse",
            action="store_true",
            help="Make the token with the heaviest weight survived.",
        )
        return parser


class CliArgsNamespaceDemo:
    """Custom subcommand implementation by user"""

    # global options
    one_line: bool
    ps1: str
    ps2: str
    verbose: int
    input_type: str
    output_type: str
    interactive: bool
    roundup = 2
    clipboard: bool
    config: str

    @classmethod
    def _append_parser(cls, parser: SaniproArgumentParser) -> None:
        """Add parser for functions included by default."""
        cls._do_append_parser(parser)

    @classmethod
    def _do_append_parser(cls, parser: SaniproArgumentParser) -> None:
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            help="Specifies a config file for each token type.",
        )
        parser.add_argument(
            "-d",
            "--input-type",
            type=str,
            choices=SupportedInTokenType.choises(),
            default="a1111compat",
            help="Preferred token type for the original prompts.",
        )
        parser.add_argument(
            "-l",
            "--one-line",
            default=False,
            action="store_true",
            help="Whether to confirm the prompt input with a single line of input.",
        )
        parser.add_argument(
            "-p",
            "--ps1",
            default=">>> ",
            type=str,
            help="The custom string that is used to wait for the user input of the prompts.",
        )
        parser.add_argument(
            "-s",
            "--output-type",
            type=str,
            choices=SupportedOutTokenType.choises(),
            default="a1111compat",
            help="Preferred token type for the processed prompts.",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            default=0,
            action="count",
            help="Switch to display the extra logs for nerds, This may be useful for debugging.  Adding more flags causes your terminal more messier.",
        )
        parser.add_argument(
            "-y",
            "--clipboard",
            default=False,
            action="store_true",
            help="Copy the result to the clipboard if possible.",
        )
        parser.add_argument(
            "-i",
            "--interactive",
            default=False,
            action="store_true",
            help="Provides the REPL interface to play with prompts. The program behaves like the Python interpreter.",
        )
        parser.add_argument(
            "--ps2",
            default="... ",
            type=str,
            help="The custom string that is used to wait for the next user input of the prompts.",
        )

    @classmethod
    def _append_subparser(cls, parser: SaniproArgumentParser) -> None:
        cls._do_append_subparser(parser)

    @classmethod
    def _do_append_subparser(cls, parser: SaniproArgumentParser) -> None:
        """User-defined parser implementation."""

    @classmethod
    def _do_get_parser(cls) -> dict:
        return {
            "prog": "sanipro",
            "description": (
                "Toolbox for Stable Diffusion prompts. "
                "'Sanipro' stands for 'pro'mpt 'sani'tizer."
            ),
            "epilog": "Help for each filter is available, respectively.",
        }

    @classmethod
    def _get_parser(cls) -> SaniproArgumentParser:
        props = cls._do_get_parser()
        return SaniproArgumentParser(
            prog=props["prog"],
            description=props["description"],
            formatter_class=SaniproHelpFormatter,
            epilog=props["epilog"],
        )

    @classmethod
    def from_sys_argv(cls, arg_val: Sequence[str]) -> tuple[Self, list[str]]:
        """Add parsers, and parse the commandline argument with it."""
        # parse global_options
        parser = cls._get_parser()
        cls._append_parser(parser)
        cls._append_subparser(parser)
        args, args_val = parser.parse_known_args(arg_val, namespace=cls())
        return args, args_val


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


from sanipro.converter_context import get_config


class CliCommandsDemo(CliCommands):
    def __init__(self, args: CliArgsNamespaceDemo, args_ret: list[str]) -> None:
        self._args = args
        self._args_ret = args_ret
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

        global_args = self._args
        args_val = self._args_ret

        # parse filter_spec
        filterpipe = FilterExecutor()
        filterpipe.append_command(RoundUpCommand(global_args.roundup))

        command_map: dict[str, type[CliCommand]] = {
            "mask": CliMaskCommand,
            "random": CliRandomCommand,
            "reset": CliResetCommand,
            "similar": CliSimilarCommand,
            "sort-all": CliSortAllCommand,
            "sort": CliSortCommand,
            "unique": CliUniqueCommand,
            "exclude": CliExcludeCommand,
        }

        class FilterId:
            name: str

        name_parser = SaniproArgumentParser(add_help=False)
        name_parser.add_argument("name", choices=command_map.keys())

        while args_val:
            arg_name, args_after_name = name_parser.parse_known_args(
                args_val, FilterId()
            )
            filter_id = arg_name.name
            command_cls = command_map[filter_id]
            curr_parser = command_cls.get_parser()

            parsed_so_far, args_val = curr_parser.parse_known_args(
                args_after_name, command_cls
            )

            inst = None
            if parsed_so_far is CliMaskCommand:
                inst = MaskCommand(parsed_so_far.mask, parsed_so_far.replace_to)
            elif parsed_so_far is CliRandomCommand:
                inst = RandomCommand(parsed_so_far.seed)
            elif parsed_so_far is CliResetCommand:
                inst = ResetCommand(parsed_so_far.value)
            elif parsed_so_far is CliSimilarCommand:
                inst = SimilarCommand(
                    parsed_so_far.method, reverse=parsed_so_far.reverse
                )
            elif parsed_so_far is CliSortAllCommand:
                inst = SortAllCommand(
                    parsed_so_far.method, reverse=parsed_so_far.reverse
                )
            elif parsed_so_far is CliSortCommand:
                inst = SortCommand(parsed_so_far.reverse)
            elif parsed_so_far is CliUniqueCommand:
                inst = UniqueCommand(parsed_so_far.reverse)
            elif parsed_so_far is CliExcludeCommand:
                inst = ExcludeCommand(parsed_so_far.exclude)

            if inst is not None:
                filterpipe.append_command(inst)

        # add filter to for converting token type
        filterpipe.append_command(
            TranslateTokenTypeCommand(self.output_type.token_type)
        )

        formatter = self._initialize_formatter(self.output_type)
        delimiter = self._initialize_delimiter(self.input_type)

        parser = self.input_type.parser(delimiter)
        token_type = self.output_type.token_type
        tokenizer = self.input_type.tokenizer(parser, token_type)

        return PromptPipelineV1(tokenizer, filterpipe, formatter)

    def _initialize_runner(self, pipe: IPromptPipeline) -> CliRunnable:
        """Returns a runner."""
        input_strategy = self._get_input_strategy()

        cli_hooks.on_init.append(prepare_readline)
        cli_hooks.execute(cli_hooks.on_init)

        if self._args.interactive:
            return RunnerFilterInteractive(
                pipe, input_strategy, PromptDifferenceDetector, self._args.clipboard
            )
        return RunnerFilterDeclarative(pipe, input_strategy)

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


def app():
    args, args_ret = CliArgsNamespaceDemo.from_sys_argv(sys.argv[1:])
    cli_commands = CliCommandsDemo(args, args_ret)

    log_level = cli_commands.get_logger_level()
    logger_root.setLevel(log_level)

    try:
        runner = cli_commands.to_runner()
    except AttributeError as e:
        logger.exception(f"error: {e}")
        exit(1)

    runner.run()


if __name__ == "__main__":
    app()
