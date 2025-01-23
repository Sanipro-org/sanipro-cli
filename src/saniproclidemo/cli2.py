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

import abc
import atexit
import logging
import os
import readline
import sys
import typing
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from sanipro.abc import IPipelineResult, IPromptPipeline
from sanipro.compatible import Self
from sanipro.converter_context import (
    A1111Config,
    Config,
    CSVConfig,
    InputConfig,
    OutputConfig,
    config_from_file,
)
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

if TYPE_CHECKING:
    from sanipro.converter_context import TokenMap


class CliCommand(abc.ABC):
    """The trait with the ability to inject a subparser."""

    @classmethod
    @abc.abstractmethod
    def get_parser(cls) -> SaniproArgumentParser:
        """Injects subparser."""


class CliExcludeCommand(CliCommand):
    exclude: Sequence[str]

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        parser = SaniproArgumentParser("exclude", formatter_class=SaniproHelpFormatter)
        parser.add_argument(
            "-x",
            "--exclude",
            nargs="*",
            help="Exclude this token from the original prompt. Multiple options can be specified.",
        )
        return parser


class CliSimilarCommand(CliCommand):
    method: ReordererStrategy
    reverse: bool

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        parser = SaniproArgumentParser(
            "similar",
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

            strategy = SequenceMatcherSimilarity()
            if method == "naive":
                return NaiveReorderer(strategy)
            elif method == "greedy":
                return GreedyReorderer(strategy)
            elif method == "kruskal":
                return KruskalMSTReorderer(strategy)
            return PrimMSTReorderer(strategy)

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
    mask: Sequence[str]
    replace_to: str

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        parser = SaniproArgumentParser(
            "mask",
            description="Mask words specified with another word (optional).",
            formatter_class=SaniproHelpFormatter,
            epilog="Note that you can still use the global `--exclude` option"
            "as well as this filter.",
        )
        parser.add_argument("mask", nargs="*", help="Masks this word.")
        parser.add_argument(
            "-t",
            "--replace-to",
            default=r"%%%",
            help="The new character or string replaced to.",
        )
        return parser


class CliRandomCommand(CliCommand):
    seed: int

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            "random",
            formatter_class=SaniproHelpFormatter,
            description="Shuffles all the prompts altogether.",
        )
        subcommand.add_argument(
            "-b", "--seed", type=int, help="Fixed randomness to this value."
        )
        return subcommand


class CliResetCommand(CliCommand):
    value: float

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            "reset",
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
    roundup: int

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            "roundup", formatter_class=SaniproHelpFormatter
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
    reverse: bool

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        subcommand = SaniproArgumentParser(
            "sort",
            formatter_class=SaniproHelpFormatter,
            description="Reorders duplicate tokens.",
            epilog="This command reorders tokens with their weights by default.",
        )
        subcommand.add_argument(
            "-r", "--reverse", action="store_true", help="With reversed order."
        )
        return subcommand


class CliSortAllCommand(CliCommand):
    reverse: bool
    method: Callable

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
            "sort-all",
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
    reverse: bool

    @classmethod
    def get_parser(cls) -> SaniproArgumentParser:
        parser = SaniproArgumentParser(
            "unique",
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
    clipboard: bool
    config: Config
    color: bool

    @classmethod
    def _append_parser(cls, parser: SaniproArgumentParser) -> None:
        """Add parser for functions included by default."""

        parser.add_argument(
            "-c",
            "--config",
            default=Config(
                A1111Config(InputConfig(","), OutputConfig(", ")),
                A1111Config(InputConfig(","), OutputConfig(", ")),
                CSVConfig(InputConfig("\n", "\t"), OutputConfig("\n", "\t")),
            ),
            type=config_from_file,
            help="Specifies a config file for each token type.",
        )

        parser.add_argument(
            "-d",
            "--input-type",
            choices=("a1111compat", "csv"),
            default="a1111compat",
            help="Preferred token type for the original prompts.",
        )

        parser.add_argument(
            "--no-color",
            action="store_false",
            default=True,
            dest="color",
            help="Without color for displaying.",
        )

        parser.add_argument(
            "-l",
            "--one-line",
            action="store_true",
            help="Whether to confirm the prompt input with a single line of input.",
        )

        parser.add_argument(
            "-p",
            "--ps1",
            default=">>> ",
            help="The custom string that is used to wait for the user input of the prompts.",
        )

        parser.add_argument(
            "-s",
            "--output-type",
            choices=("a1111", "a1111compat", "csv"),
            default="a1111compat",
            help="Preferred token type for the processed prompts.",
        )

        parser.add_argument(
            "-v", "--verbose", action="count", help="Switch to display the extra logs."
        )

        parser.add_argument(
            "-y",
            "--clipboard",
            action="store_true",
            help="Copy the result to the clipboard if possible.",
        )

        parser.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help="Provides the REPL interface to play with prompts.",
        )

        parser.add_argument(
            "--ps2",
            default="... ",
            help="The custom string that is used to wait for the next user input of the prompts.",
        )

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
            add_help=True,
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


class CliCommandsDemo(CliCommands):
    def __init__(self, args: CliArgsNamespaceDemo, args_ret: list[str]) -> None:
        self._args = args
        self._args_ret = args_ret
        self._config = self._args.config
        self.input_type = self._config.get_input_token_class(self._args.input_type)
        self.output_type = self._config.get_output_token_class(self._args.output_type)

    def _get_input_strategy(self) -> InputStrategy:
        if not self._args.interactive:
            return inputs.DirectInputStrategy()

        ps1 = self._args.ps1
        ps2 = self._args.ps2

        return (
            inputs.OnelineInputStrategy(ps1, self._args.color)
            if self._args.one_line
            else inputs.MultipleInputStrategy(ps1, ps2, self._args.color)
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
        from sanipro.pipeline_v1 import PromptPipelineV1

        args_val = self._args_ret

        # parse filter_spec
        filterpipe = FilterExecutor()

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

        def _split_help(args_ret: list[str]):
            help = []
            new_args = []

            for item in args_ret:
                if item in ("--help", "-h"):
                    help.append(item)
                    continue
                new_args.append(item)
            return new_args, help

        args_val, help_stack = _split_help(args_val)

        while args_val:
            name_parsed_success = False
            args_after_name = None
            curr_parser = None
            command_cls = None

            while not name_parsed_success:
                arg_name, args_after_name = name_parser.parse_known_args(
                    args_val, FilterId()
                )
                filter_id = arg_name.name
                command_cls = command_map[filter_id]
                curr_parser = command_cls.get_parser()

                if (
                    not args_after_name
                    and (  # entire cli arguments were so far consumed
                        help_stack  # -h/--help was specified
                    )
                ):
                    # regard "--help" was specified at the end of the cli argument
                    args_val.append(help_stack.pop())
                    continue
                # accepted
                name_parsed_success = True

            if curr_parser is None or args_after_name is None or curr_parser is None:
                raise Exception("failed to parse an argument")

            arg_parsed_success = False
            parsed_so_far = None

            while not arg_parsed_success:
                parsed_so_far, args_val = curr_parser.parse_known_args(
                    args_after_name, command_cls
                )
                if not args_val and (  # entire cli arguments were so far consumed
                    help_stack  # -h/--help was specified
                ):
                    # regard "--help" was specified at the end of the cli argument
                    args_after_name.append(help_stack.pop())
                    continue
                # accepted
                arg_parsed_success = True

            inst = None
            if parsed_so_far is CliMaskCommand:
                inst = MaskCommand(parsed_so_far.mask, parsed_so_far.replace_to)
            elif parsed_so_far is RoundUpCommand:
                inst = RoundUpCommand(parsed_so_far.roundup)
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

        pipeline = PromptPipelineV1(tokenizer, filterpipe, formatter)
        return pipeline

    def _get_runner(self) -> CliRunnable:
        pipe = self._get_pipeline()
        runner = self._initialize_runner(pipe)
        return runner

    def to_runner(self) -> CliRunnable:
        return self._get_runner()


def colorize(text: str, use_color: bool) -> typing.Any:
    if use_color:
        return style(text)

    def _inner(t: str) -> str:
        return t

    return _inner(text)


def app():
    args, args_ret = CliArgsNamespaceDemo.from_sys_argv(sys.argv[1:])

    logging.basicConfig(
        format=colorize(
            ("[%(levelname)s] %(module)s/%(funcName)s (%(lineno)d):") + " %(message)s",
            args.color,
        ),
        datefmt=r"%Y-%m-%d %H:%M:%S",
    )

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
