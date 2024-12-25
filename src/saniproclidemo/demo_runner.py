import logging
import os
import readline
import subprocess
import tempfile
import typing

from sanipro.abc import MutablePrompt
from sanipro.compatible import Self
from sanipro.diff import PromptDifferenceDetector
from sanipro.pipeline import PromptPipeline
from sanipro.promptset import SetCalculatorWrapper

from saniprocli.abc import InputStrategy, StatShowable
from saniprocli.cli_runner import (
    ExecuteDual,
    ExecuteSingle,
    RunnerInteractive,
    RunnerNonInteractive,
)

logger = logging.getLogger(__name__)


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
