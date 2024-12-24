import logging
import os
import readline
import subprocess
import tempfile
import typing

from sanipro.abc import MutablePrompt
from sanipro.compatible import Self
from sanipro.pipeline import PromptPipeline

from saniprocli.abc import InputStrategy, StatShowable
from saniprocli.cli_runner import RunnerInteractiveMultiple, RunnerInteractiveSingle

logger = logging.getLogger(__name__)


class RunnerTagFind(RunnerInteractiveSingle):
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
        with text as fp:
            try:
                for row in map(
                    lambda line: line.split(delim),
                    map(lambda ln: ln.strip("\n").replace("_", " "), fp.readlines()),
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

    def copy_to_clipboard(self, text: str) -> None:
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
                self.copy_to_clipboard(selialized)
            except subprocess.SubprocessError:
                logger.error("failed to copy to clipboard")
            except FileNotFoundError:
                logger.debug("clipboard API is not available")

        return selialized


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
        prompt_first_before = self._pipeline.tokenize(first)
        prompt_second_before = self._pipeline.tokenize(second)

        prompt = [
            self._token_cls(name=x.name, weight=x.weight)
            for x in self._calculator.do_math(prompt_first_before, prompt_second_before)
        ]

        tokens = [str(token) for token in prompt]

        self._show_cli_stat(prompt_first_before, prompt)
        self._show_cli_stat(prompt_second_before, prompt)

        selialized = self._pipeline.delimiter.sep_output.join(tokens)
        return selialized
