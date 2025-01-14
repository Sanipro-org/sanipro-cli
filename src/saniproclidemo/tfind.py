import argparse
import os
import readline
import sys
import typing
from collections.abc import Callable

from sanipro.abc import IPromptPipeline, TokenInterface
from sanipro.compatible import Self
from sanipro.converter_context import TokenMap, get_config
from sanipro.delimiter import Delimiter
from sanipro.filter_exec import FilterExecutor
from sanipro.filters.translate import TranslateTokenTypeCommand
from sanipro.logger import logger, logger_root
from sanipro.parser import NormalParser
from sanipro.pipeline_v1 import PromptPipelineV1
from sanipro.tokenizer import SimpleTokenizer

from saniprocli import inputs
from saniprocli.abc import CliRunnable, InputStrategy
from saniprocli.cli_runner import ExecuteSingle, RunnerDeclarative, RunnerInteractive
from saniprocli.commands import (
    CliArgsNamespaceDefault,
    CliCommands,
    SubparserInjectable,
)
from saniprocli.help_formatter import SaniproHelpFormatter
from saniprocli.sanipro_argparse import SaniproArgumentParser
from saniprocli.textutils import (
    ClipboardHandler,
    CSVUtilsBase,
    dump_to_file,
    get_temp_filename,
)
from saniproclidemo.cli import StatisticsHandler


class CliArgsNamespaceDemo(CliArgsNamespaceDefault):
    # global options
    interactive: bool

    # 'dest' name for general operations
    operation_id = str  # may be 'filter', 'set_op', and more

    # for tfind subcommand
    infile: typing.TextIO
    key_field: int
    value_field: int
    field_delimiter: str
    tempdir: str

    clipboard: bool
    config: str

    def is_tfind(self) -> bool:
        return self.operation_id == CliSubcommandSearchTag.command_id

    @classmethod
    def _do_append_parser(cls, parser: SaniproArgumentParser) -> None:
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

        classes: list[type[SubparserInjectable]] = [CliSubcommandSearchTag]

        for cmd in classes:
            cmd.inject_subparser(subparser)


class CsvUtils(CSVUtilsBase):
    def _do_preprocess(self, column: list[str]):
        column[0] = self.replace_underscore(column[0])
        return column

    def replace_underscore(self, line: str) -> str:
        line = line.strip()
        # line = line.replace("_", " ")
        return line


class RunnerTagFindInteractive(ExecuteSingle, RunnerInteractive):
    """Represents the runner specialized for the filtering mode."""

    _histfile: str

    def __init__(
        self,
        pipeline: IPromptPipeline,
        tags_n_count: dict[str, str],
        strategy: InputStrategy,
        tempdir: str,
        use_clipboard: bool,
    ) -> None:
        self._app = pipeline
        self._tokenizer = pipeline.tokenizer
        self._input_strategy = strategy
        self.tags_n_count: dict[str, str] = tags_n_count
        self.tempdir = tempdir
        self._use_clipboard = use_clipboard

    @classmethod
    def create_from_csv(
        cls,
        pipeline: IPromptPipeline,
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
        utilities assume the field index originates from 1.

        The `tempdir` is used to store the history file for the GNU Readline.
        """

        try:
            tag_n_count = CsvUtils.create_dict_from_io(text, delim, key_idx, value_idx)
            return cls(pipeline, tag_n_count, strategy, tempdir, use_clipboard)
        except IndexError as e:
            raise type(e)

    def _on_init(self) -> None:
        histfile = get_temp_filename(self.tempdir)
        dump_to_file(histfile, self.tags_n_count.keys())

        try:
            readline.read_history_file(histfile)
        except FileNotFoundError as e:
            raise type(e)("failed to read history file: %s" % (histfile,))

        # so that the file is deleted after the program exits
        self._histfile = histfile

    def _on_exit(self) -> None:
        try:
            os.remove(self._histfile)
        except OSError:
            logger.warning("%s: history file was not deleted", self._histfile)

    def _execute_single_inner(self, source: str) -> str:
        result = self._app.execute(source)
        StatisticsHandler.show_cli_stat(result)
        selialized = str(self._app)
        if self._use_clipboard:
            ClipboardHandler.copy_to_clipboard(selialized)
        return selialized


class RunnerTagFindNonInteractive(ExecuteSingle, RunnerDeclarative):
    """Represents the runner specialized for the filtering mode."""

    def __init__(
        self,
        pipeline: IPromptPipeline,
        tags_n_count: dict[str, str],
        strategy: InputStrategy,
    ) -> None:
        self._app = pipeline
        self._tokenizer = pipeline.tokenizer
        self._input_strategy = strategy
        self.tags_n_count: dict[str, str] = tags_n_count
        self._histfile = ""

    @classmethod
    def create_from_csv(
        cls,
        pipeline: IPromptPipeline,
        text: typing.TextIO,
        strategy: InputStrategy,
        delim: str,
        key_idx: int,
        value_idx: int,
    ) -> Self:
        """Import the key-value storage from a comma-separated file.
        The index starts from 1. This is because common traditional command-line
        utilities assume the field index originates from 1."""
        try:
            tags_n_count = CsvUtils.create_dict_from_io(text, delim, key_idx, value_idx)
            return cls(pipeline, tags_n_count, strategy)
        except IndexError:
            raise

    def _execute_single_inner(self, source: str) -> str:
        result = self._app.execute(source)
        StatisticsHandler.show_cli_stat(result)
        selialized = str(self._app)
        return selialized


class DummyParser(NormalParser):
    def get_token(
        self, sentence: str, token_cls: type[TokenInterface]
    ) -> typing.Generator[TokenInterface, None, None]:
        return (
            token_cls(element.strip(), 1.0)
            for element in sentence.split(self._delimiter.sep_input)
        )


def format_tfind_token(token: TokenInterface) -> str:
    return token.name


class CliCommandsDemo(CliCommands):
    def __init__(self, args: CliArgsNamespaceDemo) -> None:
        self._args = args
        self._config = get_config(self._args.config)

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

    def _initialize_formatter(self, token_map: TokenMap) -> Callable:
        """Initialize formatter function which takes an only 'Token' class.
        Note when 'csv' is chosen as Token, the token_map.formatter is
        a partial function."""

        formatter = token_map.formatter

        if token_map.type_name == "csv":
            new_formatter = formatter(token_map.field_separator)
            formatter = new_formatter

        return formatter

    def _initialize_delimiter(self, itype: TokenMap) -> Delimiter:
        its = self._config.get_input_token_separator(self._args.input_type)
        ots = self._config.get_output_token_separator(self._args.output_type)
        ifs = itype.field_separator

        delimiter = Delimiter(its, ots, ifs)
        return delimiter

    def _initialize_pipeline(self) -> IPromptPipeline:
        itype = self._config.get_input_token_class(self._args.input_type)
        otype = self._config.get_output_token_class(self._args.output_type)

        formatter = self._initialize_formatter(otype)
        filter_pipe = self._initialize_filter_pipeline()
        delimiter = self._initialize_delimiter(itype)

        if self._args.is_tfind():
            parser = DummyParser(delimiter)
            token_type = itype.token_type
            tokenizer = SimpleTokenizer(parser, token_type)
            return PromptPipelineV1(tokenizer, filter_pipe, formatter)
        raise

    def _initialize_filter_pipeline(self) -> FilterExecutor:
        filterpipe = FilterExecutor()

        # add filter to for converting token type
        token_type = self._config.get_output_token_class(self._args.output_type)
        filterpipe.append_command(TranslateTokenTypeCommand(token_type.token_type))

        return filterpipe

    def _initialize_runner(self, pipe: IPromptPipeline) -> CliRunnable:
        """Returns a runner."""
        input_strategy = self._get_input_strategy()

        if self._args.is_tfind():
            if self._args.interactive:
                return RunnerTagFindInteractive.create_from_csv(
                    pipe,
                    self._args.infile,
                    input_strategy,
                    self._args.field_delimiter,
                    self._args.key_field,
                    self._args.value_field,
                    self._args.tempdir,
                    self._args.clipboard,
                )

            return RunnerTagFindNonInteractive.create_from_csv(
                pipe,
                self._args.infile,
                input_strategy,
                self._args.field_delimiter,
                self._args.key_field,
                self._args.value_field,
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
