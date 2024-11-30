import argparse
import logging
import pprint
from collections.abc import Sequence

from sanipro import common
from sanipro.parser import TokenInteractive, TokenNonInteractive
from sanipro.utils import HasPrettyRepr

from saniprocli import color, inputs
from saniprocli.abc import RunnerInterface
from saniprocli.cli_runner import RunnerInteractive, RunnerNonInteractive

from .help_formatter import SaniproHelpFormatter
from .utils import get_debug_fp, get_log_level_from

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class CommandsBase(HasPrettyRepr):
    input_delimiter = ","
    interactive = False
    one_line = False
    output_delimiter = ", "
    ps1 = f"\001{color.CYAN}\002>>>\001{color.RESET}\002 "
    ps2 = f"\001{color.CYAN}\002...\001{color.RESET}\002 "

    filter: str | None = None
    verbose: int | None = None

    def get_logger_level(self) -> int:
        if self.verbose is None:
            return logging.WARNING
        try:
            log_level = get_log_level_from(self.verbose)
            return log_level
        except ValueError:
            raise ValueError("the maximum two -v flags can only be added")

    def debug(self) -> None:
        pprint.pprint(self, get_debug_fp())

    @classmethod
    def prepare_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="sanipro",
            description=(
                "Toolbox for Stable Diffusion prompts. "
                "'Sanipro' stands for 'pro'mpt 'sani'tizer."
            ),
            formatter_class=SaniproHelpFormatter,
            epilog="Help for each filter is available, respectively.",
        )

        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            help=(
                "Switch to display the extra logs for nerds, "
                "This may be useful for debugging."
                "Adding more flags causes your terminal more messier."
            ),
        )

        parser.add_argument(
            "-d",
            "--input-delimiter",
            type=str,
            default=cls.input_delimiter,
            help=("Preferred delimiter string for the original prompts. " ""),
        )

        parser.add_argument(
            "-s",
            "--output-delimiter",
            default=cls.output_delimiter,
            type=str,
            help=("Preferred delimiter string for the processed prompts. " ""),
        )

        parser.add_argument(
            "-p",
            "--ps1",
            default=cls.ps1,
            type=str,
            help=(
                "The custom string that is used to wait for the user input "
                "of the prompts."
            ),
        )

        parser.add_argument(
            "--ps2",
            default=cls.ps2,
            type=str,
            help=(
                "The custom string that is used to wait for the next user "
                "input of the prompts."
            ),
        )

        parser.add_argument(
            "-i",
            "--interactive",
            default=cls.interactive,
            action="store_true",
            help=(
                "Provides the REPL interface to play with prompts. "
                "The program behaves like the Python interpreter."
            ),
        )

        parser.add_argument(
            "-l",
            "--one-line",
            default=cls.one_line,
            action="store_true",
            help=("Whether to confirm the prompt input with a single line of input."),
        )

        # This creates the global parser.
        cls.append_parser(parser)

        # This creates the user-defined subparser.
        cls.append_subparser(parser)

        return parser

    def to_runner(self) -> RunnerInterface:
        pipeline = self.get_pipeline()
        runner = None
        strategy = None

        if self.one_line:
            strategy = inputs.OnelineInputStrategy(self.ps1)
        else:
            strategy = inputs.MultipleInputStrategy(self.ps1, self.ps2)

        if self.interactive:
            runner = RunnerInteractive(pipeline, TokenInteractive, strategy)
        else:
            runner = RunnerNonInteractive(pipeline, TokenNonInteractive, strategy)
        return runner

    def get_pipeline(self) -> common.PromptPipeline: ...

    @classmethod
    def append_parser(cls, parser: argparse.ArgumentParser): ...

    @classmethod
    def append_subparser(cls, parser: argparse.ArgumentParser): ...

    @classmethod
    def from_sys_argv(cls, arg_val: Sequence) -> "CommandsBase":
        parser = cls.prepare_parser()
        args = parser.parse_args(arg_val, namespace=cls())

        return args
