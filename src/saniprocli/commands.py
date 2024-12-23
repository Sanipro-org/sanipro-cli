import argparse
import logging
from collections.abc import Sequence

from sanipro.compatible import Self
from sanipro.parser import TokenInteractive, TokenNonInteractive
from sanipro.pipeline import PromptPipeline
from sanipro.utils import HasPrettyRepr

from saniprocli import inputs
from saniprocli.abc import (
    CliRunnable,
    ParserAppendable,
    PipelineGettable,
    SubParserAppendable,
)
from saniprocli.cli_runner import RunnerInteractiveSingle, RunnerNonInteractiveSingle

from .help_formatter import SaniproHelpFormatter
from .logger import get_log_level_from

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class CliArgsNamespaceDefault(HasPrettyRepr, ParserAppendable, SubParserAppendable):
    """Default namespace for the argparser."""

    input_delimiter: str
    interactive: bool
    one_line: bool
    output_delimiter: str
    ps1: str
    ps2: str
    verbose: int

    @classmethod
    def _append_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Add parser for functions included by default."""

        parser.add_argument(
            "-d",
            "--input-delimiter",
            type=str,
            default=",",
            help=("Preferred delimiter string for the original prompts."),
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
            "-l",
            "--one-line",
            default=False,
            action="store_true",
            help=("Whether to confirm the prompt input with a single line of input."),
        )

        parser.add_argument(
            "-s",
            "--output-delimiter",
            default=", ",
            type=str,
            help=("Preferred delimiter string for the processed prompts."),
        )

        parser.add_argument(
            "-p",
            "--ps1",
            default=">>> ",
            type=str,
            help=(
                "The custom string that is used to wait for the user input "
                "of the prompts."
            ),
        )

        parser.add_argument(
            "--ps2",
            default="... ",
            type=str,
            help=(
                "The custom string that is used to wait for the next user "
                "input of the prompts."
            ),
        )

        parser.add_argument(
            "-v",
            "--verbose",
            default=0,
            action="count",
            help=(
                "Switch to display the extra logs for nerds, "
                "This may be useful for debugging. "
                "Adding more flags causes your terminal more messier."
            ),
        )
        cls._do_append_parser(parser)

    @classmethod
    def _do_append_parser(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @classmethod
    def _append_subparser(cls, parser: argparse.ArgumentParser) -> None:
        cls._do_append_subparser(parser)

    @classmethod
    def _do_append_subparser(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @classmethod
    def from_sys_argv(cls, arg_val: Sequence[str]) -> Self:
        """Add parsers, and parse the commandline argument with it."""

        parser = argparse.ArgumentParser(
            prog="sanipro",
            description=(
                "Toolbox for Stable Diffusion prompts. "
                "'Sanipro' stands for 'pro'mpt 'sani'tizer."
            ),
            formatter_class=SaniproHelpFormatter,
            epilog="Help for each filter is available, respectively.",
        )

        cls._append_parser(parser)
        cls._append_subparser(parser)

        args = parser.parse_args(arg_val, namespace=cls())
        return args


class CliCommands(PipelineGettable):
    def __init__(self, args: CliArgsNamespaceDefault):
        self._args = args

    def get_logger_level(self) -> int:
        if self._args.verbose is None:
            return logging.WARNING
        try:
            log_level = get_log_level_from(self._args.verbose)
            return log_level
        except ValueError:
            raise ValueError("the maximum two -v flags can only be added")

    def to_runner(self) -> CliRunnable:
        """The default factory method for Runner class.

        Instantiated instance will be switched by the command option."""

        pipe = self._get_pipeline()
        ps1 = self._args.ps1
        ps2 = self._args.ps2

        strategy = (
            inputs.OnelineInputStrategy(ps1)
            if self._args.one_line
            else inputs.MultipleInputStrategy(ps1, ps2)
        )
        runner = (
            RunnerInteractiveSingle(pipe, TokenInteractive, strategy)
            if self._args.interactive
            else RunnerNonInteractiveSingle(pipe, TokenNonInteractive, strategy)
        )

        return runner

    def _get_pipeline(self) -> PromptPipeline:
        raise NotImplementedError
