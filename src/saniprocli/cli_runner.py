import logging
import sys
import time

from sanipro.abc import MutablePrompt, TokenInterface
from sanipro.diff import PromptDifferenceDetector
from sanipro.pipeline import PromptPipeline
from sanipro.promptset import SetCalculatorWrapper

from saniprocli import cli_hooks, color
from saniprocli.abc import (
    CliRunnable,
    CliRunnableInnerRun,
    InputStrategy,
)

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class RunnerInteractive(CliRunnable, CliRunnableInnerRun):
    """Represents the method for the program to interact
    with the users.

    This runner is used when the user decided to use
    the interactive mode. This is similar what Python interpreter does like."""

    def _start_loop(self) -> None:
        raise NotImplementedError

    def _try_banner(self) -> None:
        """TODO implement an option whether to show the banner or not"""
        self._write(
            f"Sanipro (created by iigau) in interactive mode\n"
            f"Program was launched up at {time.asctime()}.\n"
        )

    def run(self) -> None:
        cli_hooks.execute(cli_hooks.on_interactive)
        self._write = sys.stdout.write
        self._try_banner()
        self._start_loop()


class RunnerInteractiveSingle(RunnerInteractive):
    def __init__(
        self,
        pipeline: PromptPipeline,
        token_cls: type[TokenInterface],
        strategy: InputStrategy,
    ) -> None:
        self._pipeline = pipeline
        self._token_cls = token_cls
        self._input_strategy = strategy

        self._detector_cls = PromptDifferenceDetector

    def _show_cli_stat(self, before: MutablePrompt, after: MutablePrompt) -> None:
        """Explains what has changed in the unprocessed/processsed prompts"""
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

    def _execute(self, source: str) -> str:
        unparsed = self._pipeline.parse(source, self._token_cls, auto_apply=True)
        parsed = self._pipeline.tokens

        self._show_cli_stat(unparsed, parsed)

        selialized = str(self._pipeline)
        return selialized

    def _start_loop(self) -> None:
        self._write = sys.stdout.write

        while True:
            try:
                try:
                    prompt_input = self._input_strategy.input()
                    if prompt_input:
                        out = self._execute(prompt_input)
                        self._write(f"{out}\n")
                except EOFError as e:
                    break
            except ValueError as e:  # like unclosed parentheses
                logger.fatal(f"error: {e}")
            except KeyboardInterrupt:
                self._write("\nKeyboardInterrupt\n")
        self._write(f"\n")


class RunnerInteractiveMultiple(RunnerInteractive):
    """TODO RunnerInterfaceに準拠させる"""

    def __init__(
        self,
        pipeline: PromptPipeline,
        token_cls: type[TokenInterface],
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
    ) -> None:
        self._pipeline = pipeline
        self._token_cls = token_cls
        self._input_strategy = strategy

        self._detector_cls = PromptDifferenceDetector
        self._calculator = calculator

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

    def _execute(self, first: str, second: str) -> str:
        prompt_first = self._pipeline.parse(first, self._token_cls)
        prompt_second = self._pipeline.parse(second, self._token_cls)

        tokens_raw = [
            self._token_cls(name=x.name, weight=x.weight)
            for x in self._calculator.do_math(prompt_first, prompt_second)
        ]

        selialized = [str(token) for token in tokens_raw]

        self._show_cli_stat(prompt_first, tokens_raw)
        self._show_cli_stat(prompt_second, tokens_raw)

        selialized = self._pipeline.delimiter.sep_output.join(selialized)
        return selialized

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
                            state = 00  # stateを00に戻す
                            color.color_foreground = _color
                            continue
                    elif state == 20:
                        out = self._execute(first, second)
                        self._write(f"{out}\n")
                        color.color_foreground = _color
                        break  # 次のプロンプトの組の入力へ
            except EOFError:
                break
        self._write(f"\n")


class RunnerNonInteractiveSingle(CliRunnable):
    """Represents the method for the program to interact
    with the users in non-interactive mode.

    Intended the case where the users feed the input from STDIN.
    """

    def __init__(
        self,
        pipeline: PromptPipeline,
        token_cls: type[TokenInterface],
        strategy: InputStrategy,
    ) -> None:
        self._pipeline = pipeline
        self._token_cls = token_cls
        self._input_strategy = strategy

    def _run_once(self) -> None:
        sentence = None
        try:
            sentence = self._input_strategy.input().strip()
        except (KeyboardInterrupt, EOFError):
            sys.stderr.write("\n")
            sys.exit(1)
        finally:
            if sentence is not None:
                self._pipeline.parse(str(sentence), self._token_cls, auto_apply=True)
                result = str(self._pipeline)
                print(result)

    def run(self) -> None:
        self._run_once()
