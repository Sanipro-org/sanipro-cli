import argparse
import atexit
import logging
import os
import readline
import sys
import typing
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from sanipro.abc import IPipelineResult, IPromptPipeline, MutablePrompt
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
from sanipro.filters.exclude import ExcludeCommand
from sanipro.filters.roundup import RoundUpCommand
from sanipro.filters.translate import TranslateTokenTypeCommand
from sanipro.logger import logger, logger_root
from sanipro.promptset import (
    DifferenceCalculator,
    IntersectionCalculator,
    SetCalculator,
    SymmetricDifferenceCalculator,
    UnionCalculator,
)

from saniprocli import cli_hooks, inputs
from saniprocli.abc import CliRunnable, InputStrategy, RunnerSetOperation
from saniprocli.cli_runner import (
    ExecuteMultiple,
    ExecuteMultipleColor,
    ExecuteMultipleNocolor,
    ExecuteSingle,
    RunnerDeclarative,
    RunnerInteractive,
)
from saniprocli.color import style
from saniprocli.commands import (
    CliArgsNamespaceDefault,
    CliCommands,
    SubparserInjectable,
)
from saniprocli.help_formatter import SaniproHelpFormatter
from saniprocli.sanipro_argparse import SaniproArgumentParser
from saniprocli.textutils import ClipboardHandler

if TYPE_CHECKING:
    from sanipro.converter_context import TokenMap

logging.basicConfig(
    format=style(
        ("[%(levelname)s] %(module)s/%(funcName)s (%(lineno)d):") + " %(message)s"
    ),
    datefmt=r"%Y-%m-%d %H:%M:%S",
)


class CliSubcommandUnion(SubparserInjectable):
    command_id: str = "union"

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id, formatter_class=SaniproHelpFormatter
        )


class CliSubcommandIntersection(SubparserInjectable):
    command_id: str = "inter"

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id, formatter_class=SaniproHelpFormatter
        )


class CliSubcommandSetDifference(SubparserInjectable):
    command_id: str = "diff"

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id, formatter_class=SaniproHelpFormatter
        )


class CliSubcommandSymmetricDifference(SubparserInjectable):
    command_id: str = "symdiff"

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        parser = subparser.add_parser(
            name=cls.command_id, formatter_class=SaniproHelpFormatter
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
    color: bool

    # 'dest' name for general operations
    operation_id: str

    clipboard: bool
    fixed_prompt: typing.TextIO
    config: Config
    color: bool

    reverse: bool

    @classmethod
    def _do_append_parser(cls, parser: SaniproArgumentParser) -> None:
        parser.add_argument(
            "-d",
            "--input-type",
            choices=("a1111compat", "csv"),
            default="a1111compat",
            help="Preferred token type for the original prompts.",
        )

        parser.add_argument(
            "-s",
            "--output-type",
            choices=("a1111", "a1111compat", "csv"),
            default="a1111compat",
            help="Preferred token type for the processed prompts.",
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
            "--color", action="store_true", help="Uses color for displaying."
        )

        parser.add_argument(
            "-x",
            "--exclude",
            nargs="*",
            help=(
                "Exclude this token from the original prompt. "
                "Multiple options can be specified."
            ),
        )

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
            "--no-color",
            action="store_false",
            default=True,
            dest="color",
            help="Without color for displaying.",
        )

        parser.add_argument(
            "-a",
            "--fixed-prompt",
            type=argparse.FileType("r"),
            help="Feed the elements of set a from a file or stdin.",
        )

        parser.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help=(
                "Provides the REPL interface to play with prompts. "
                "The program behaves like the Python interpreter."
            ),
        )

        parser.add_argument(
            "-y",
            "--clipboard",
            action="store_true",
            help="Copy the result to the clipboard if possible.",
        )

        parser.add_argument(
            "-r", "--reverse", action="store_true", help="reverse two sets"
        )

    @classmethod
    def _do_append_subparser(cls, parser: SaniproArgumentParser) -> None:
        subparser = parser.add_subparsers(title="operations", dest="operation_id")

        classes: list[type[SubparserInjectable]] = [
            CliSubcommandUnion,
            CliSubcommandIntersection,
            CliSubcommandSetDifference,
            CliSubcommandSymmetricDifference,
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


def RunnerSetOperationInteractiveDual(use_color: bool, *args, **kwargs):
    ColorMixin: type[ExecuteMultiple] = (
        ExecuteMultipleColor if use_color else ExecuteMultipleNocolor
    )

    class _RunnerSetOperationInteractiveDual(ColorMixin, RunnerInteractive, RunnerSetOperation):  # type: ignore
        def __init__(
            self,
            pipeline: IPromptPipeline,
            strategy: InputStrategy,
            calculator: SetCalculator,
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
                for x in self._calculator.do_math(
                    prompt_first_before, prompt_second_before
                )
            ]

            logger.info("(statistics) prompt A <> result")
            StatisticsHandler.show_cli_stat(PipelineResult(prompt_first_before, prompt))

            logger.info("(statistics) prompt B <> result")
            StatisticsHandler.show_cli_stat(
                PipelineResult(prompt_second_before, prompt)
            )

            selialized = str(self._pipeline.new(prompt))
            logger.debug(self._pipeline.new)

            if self._use_clipboard:
                ClipboardHandler.copy_to_clipboard(selialized)

            return selialized

    return _RunnerSetOperationInteractiveDual(*args, **kwargs)


class RunnerSetOperationDeclarativeMono(
    ExecuteSingle, RunnerDeclarative, RunnerSetOperation
):
    def __init__(
        self,
        pipeline: IPromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculator,
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
        calculator: SetCalculator,
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

        self._args = args
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
        filterpipe.append_command(RoundUpCommand(self._args.roundup))

        if self._args.exclude:
            filterpipe.append_command(ExcludeCommand(self._args.exclude))

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

        def create_from(key: str | None = "union") -> SetCalculator:
            """Creates the instance from the key"""
            if key == "inter":
                return IntersectionCalculator()
            elif key == "diff":
                return DifferenceCalculator(reverse=self._args.reverse)
            elif key == "symdiff":
                return SymmetricDifferenceCalculator()

            return UnionCalculator()

        calculator = create_from(self._args.operation_id)
        if self._args.interactive:
            return RunnerSetOperationInteractiveDual(
                self._args.color,
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

    def _get_pipeline(self) -> IPromptPipeline:
        pipeline_cls = self._initialize_pipeline()
        return pipeline_cls

    def _get_runner(self) -> CliRunnable:
        pipe = self._get_pipeline()
        runner = self._initialize_runner(pipe)
        return runner

    def to_runner(self) -> CliRunnable:
        return self._get_runner()


def app():
    args = CliArgsNamespaceDemo.from_sys_argv(sys.argv[1:])
    cli_commands = CliCommandsDemo(args)

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
