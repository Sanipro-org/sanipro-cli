import logging
import os
import readline
import sys
import tempfile
import time
import typing
from abc import ABC, abstractmethod

from sanipro.abc import MutablePrompt
from sanipro.diff import PromptDifferenceDetector
from sanipro.pipeline import PromptPipeline
from sanipro.promptset import SetCalculatorWrapper

from saniprocli import cli_hooks, color
from saniprocli.abc import (
    CliPlural,
    CliRunnable,
    CliRunnableInnerRun,
    CliSingular,
    InputStrategy,
    StatShowable,
)

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class RunnerInteractive(CliRunnable, CliRunnableInnerRun, ABC):
    """Represents the method for the program to interact
    with the users.

    This runner is used when the user decided to use
    the interactive mode.

    This is similar what Python interpreter does like."""

    @abstractmethod
    def _start_loop(self) -> None:
        """The actual start of the interaction with the user."""

    def _try_banner(self) -> None:
        """Tries to show the banner if possible,

        TODO implement an option whether to show the banner or not."""
        self._write(
            f"Sanipro (created by iigau) in interactive mode\n"
            f"Program was launched up at {time.asctime()}.\n"
        )

    def run(self) -> None:
        cli_hooks.execute(cli_hooks.on_interactive)
        self._write = sys.stdout.write
        self._try_banner()
        self._start_loop()


class RunnerInteractiveSingle(RunnerInteractive, CliSingular):
    """Represents the runner with the interactive user interface
    that expects a single input of the prompt."""

    def __init__(self, pipeline: PromptPipeline, strategy: InputStrategy) -> None:
        self._pipeline = pipeline
        self._token_cls = pipeline.token_cls
        self._input_strategy = strategy

        self._detector_cls = PromptDifferenceDetector

    def _execute_single_inner(self, source: str) -> str:
        return source

    def _execute_single(self, source: str) -> str:
        return self._execute_single_inner(source)

    def _start_loop(self) -> None:
        self._write = sys.stdout.write

        while True:
            try:
                try:
                    prompt_input = self._input_strategy.input()
                    if prompt_input:
                        out = self._execute_single(prompt_input)
                        self._write(f"{out}\n")
                except EOFError as e:
                    break
            except ValueError as e:  # like unclosed parentheses
                logger.fatal(f"error: {e}")
            except KeyboardInterrupt:
                self._write("\nKeyboardInterrupt\n")
        self._write(f"\n")


class RunnerInteractiveMultiple(RunnerInteractive, CliPlural):
    """Represents the runner with the interactive user interface
    that expects two different prompts."""

    def __init__(
        self,
        pipeline: PromptPipeline,
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
    ) -> None:
        self._pipeline = pipeline
        self._token_cls = pipeline.token_cls
        self._input_strategy = strategy

        self._detector_cls = PromptDifferenceDetector
        self._calculator = calculator

    def _execute_multi_inner(self, first: str, second: str) -> str:
        raise NotImplementedError

    def _execute_multi(self, first: str, second: str) -> str:
        return self._execute_multi_inner(first, second)

    def _handle_input(self) -> str:
        while True:
            try:
                prompt_input = self._input_strategy.input()
                if prompt_input:
                    return prompt_input
            except EOFError as e:
                raise EOFError("EOF received. Going back to previous state.")
            except KeyboardInterrupt:
                self._write("\nKeyboardInterrupt\n")
            except Exception as e:
                logger.fatal(f"error: {e}")

    def _start_loop(self) -> None:
        self._write = sys.stdout.write

        while True:
            try:
                state = 00
                first = ""
                second = ""
                _color = color.color_foreground

                while True:
                    if state == 00:
                        try:
                            first = self._handle_input()
                            if first:
                                state = 10
                        except EOFError:
                            raise
                    elif state == 10:
                        color.color_foreground = "green"
                        try:
                            second = self._handle_input()
                            if second:
                                state = 20
                        except EOFError:
                            state = 00  # reset state to 00
                            color.color_foreground = _color
                            continue
                    elif state == 20:
                        out = self._execute_multi(first, second)
                        self._write(f"{out}\n")
                        color.color_foreground = _color
                        break  # go to next set of prompts
            except EOFError:
                break
        self._write(f"\n")


class RunnerNonInteractiveSingle(CliRunnable, CliSingular):
    """Represents the method for the program to interact
    with the users in non-interactive mode.

    Intended the case where the users feed the input from STDIN.
    """

    def __init__(self, pipeline: PromptPipeline, strategy: InputStrategy) -> None:
        self._pipeline = pipeline
        self._token_cls = pipeline.token_cls
        self._input_strategy = strategy

    def _execute_single(self, source: str) -> str:
        self._pipeline.tokenize(str(source))
        return str(self._pipeline)

    def _run_once(self) -> None:
        self._write = print
        sentence = ""
        try:
            sentence = self._input_strategy.input().strip()
        except (KeyboardInterrupt, EOFError):
            sys.stderr.write("\n")
            sys.exit(1)
        finally:
            out = self._execute_single(sentence)
            self._write(out)

    def run(self) -> None:
        self._run_once()


class RunnerFilter(RunnerInteractiveSingle, StatShowable):
    """Represents the runner specialized for the filtering mode."""

    def _show_cli_stat(self, before: MutablePrompt, after: MutablePrompt) -> None:
        detector = self._detector_cls(before, after)
        items = [
            f"before -> {detector.before_num}",
            f"after -> {detector.after_num}",
            f"reduced -> {detector.reduced_num}",
        ]

        if detector.duplicated:
            del_string = ", ".join(detector.duplicated)
            items.append(f"duplicated -> {del_string}")
        else:
            items.append("no duplicates detected")

        for item in items:
            logger.info(f"(statistics) {item}")

    def _execute_single_inner(self, source: str) -> str:
        unparsed = self._pipeline.tokenize(source)
        parsed = self._pipeline.execute(source)

        self._show_cli_stat(unparsed, parsed)

        selialized = str(self._pipeline)
        return selialized


class RunnerSetOperation(RunnerInteractiveMultiple, StatShowable):
    """Represents the runner specialized for the set operation."""

    def _show_cli_stat(self, before: MutablePrompt, after: MutablePrompt) -> None:
        detector = self._detector_cls(before, after)
        items = [
            f"before -> {detector.before_num}",
            f"after -> {detector.after_num}",
            f"reduced -> {detector.reduced_num}",
        ]

        if detector.duplicated:
            del_string = ", ".join(detector.duplicated)
            items.append(f"duplicated -> {del_string}")
        else:
            items.append("no duplicates detected")

        for item in items:
            logger.info(f"(statistics) {item}")

    def _execute_multi_inner(self, first: str, second: str) -> str:
        prompt_first = self._pipeline.tokenize(first)
        prompt_second = self._pipeline.tokenize(second)

        tokens_raw = [
            self._token_cls(name=x.name, weight=x.weight)
            for x in self._calculator.do_math(prompt_first, prompt_second)
        ]

        tokens = [str(token) for token in tokens_raw]

        self._show_cli_stat(prompt_first, tokens_raw)
        self._show_cli_stat(prompt_second, tokens_raw)

        selialized = self._pipeline.delimiter.sep_output.join(tokens)
        return selialized


class RunnerTagFind(RunnerInteractiveSingle):
    """Represents the runner specialized for the filtering mode."""

    _csvfile: typing.TextIO

    def __init__(self, csvfile: typing.TextIO, strategy: InputStrategy) -> None:
        self._input_strategy = strategy
        self._csvfile = csvfile

        self.column_separator = ","
        self.tags_n_count: dict[str, str] = {}

        with self._csvfile as fp:
            for line in map(lambda ln: ln.strip("\n"), fp.readlines()):
                key, value = line.split(self.column_separator)
                self.tags_n_count |= {key: value}

    def _create_histfile_at_dev_shm(self) -> str:
        with tempfile.NamedTemporaryFile(delete=False, dir="/dev/shm") as fp:
            histfile = fp.name
            for key in self.tags_n_count.keys():
                fp.write(f"{key}\n".encode())
        return histfile

    def _prepare_readline_with_histfile(self, histfile: str) -> None:
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            logger.exception(f"failed to read history file: {histfile}")

    def _start_loop(self) -> None:
        histfile = self._create_histfile_at_dev_shm()
        self._prepare_readline_with_histfile(histfile)

        self._write = sys.stdout.write

        while True:
            try:
                try:
                    prompt_input = self._input_strategy.input()
                    if prompt_input:
                        out = self._execute_single(prompt_input)
                        self._write(f"{out}\n")
                except EOFError as e:
                    break
            except ValueError as e:  # like unclosed parentheses
                logger.fatal(f"error: {e}")
            except KeyboardInterrupt:
                self._write("\nKeyboardInterrupt\n")
        self._write(f"\n")

        os.remove(histfile)

    def _execute_single_inner(self, source: str) -> str:
        answer = self.tags_n_count.get(source, "no tag found")
        return answer
