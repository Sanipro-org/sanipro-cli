import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from sanipro.abc import IPromptPipeline
from sanipro.compatible import Self
from sanipro.utils import HasPrettyRepr

from saniprocli.abc import (
    CliRunnable,
    ParserAppendable,
    PipelineGettable,
    SubParserAppendable,
)
from saniprocli.help_formatter import SaniproHelpFormatter
from saniprocli.logger import get_log_level_from
from saniprocli.sanipro_argparse import SaniproArgumentParser


class CliArgsNamespaceDefault(HasPrettyRepr, ParserAppendable, SubParserAppendable):
    """Default namespace for the argparser."""

    one_line: bool
    ps1: str
    ps2: str
    verbose: int

    @classmethod
    def _append_parser(cls, parser: SaniproArgumentParser) -> None:
        """Add parser for functions included by default."""

        parser.add_argument(
            "-l",
            "--one-line",
            default=False,
            action="store_true",
            help=("Whether to confirm the prompt input with a single line of input."),
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
    def _do_append_parser(cls, parser: SaniproArgumentParser) -> None:
        """User-defined parser implementation."""

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
    def from_sys_argv(cls, arg_val: Sequence[str]) -> Self:
        """Add parsers, and parse the commandline argument with it."""

        parser = cls._get_parser()

        cls._append_parser(parser)
        cls._append_subparser(parser)

        args = parser.parse_args(arg_val, namespace=cls())
        return args


class CliCommands(PipelineGettable, ABC):
    def __init__(self, args: CliArgsNamespaceDefault):
        self._args = args

    def get_logger_level(self) -> int:
        if self._args.verbose == 0:
            return logging.WARNING
        try:
            log_level = get_log_level_from(self._args.verbose)
            return log_level
        except ValueError:
            raise ValueError("the maximum two -v flags can only be added")

    @abstractmethod
    def to_runner(self) -> CliRunnable:
        """The default factory method for Runner class.

        Instantiated instance will be switched by the command option."""

    def _get_pipeline(self) -> IPromptPipeline:
        raise NotImplementedError


class SubparserInjectable(ABC):
    """The trait with the ability to inject a subparser."""

    @classmethod
    @abstractmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        """Injects subparser."""


class CliCommand(SubparserInjectable):
    """The wrapper class for the filter commands
    with the addition of subparser."""

    command_id: str

    @classmethod
    def inject_subparser(cls, subparser: argparse._SubParsersAction) -> None:
        """Does nothing by default."""
