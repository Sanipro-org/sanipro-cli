import logging
import sys
import time

from sanipro.abc import TokenInterface
from sanipro.diff import PromptDifferenceDetector
from sanipro.pipeline import PromptPipeline

from saniprocli import cli_hooks
from saniprocli.abc import InputStrategy, RunnerInterface

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


def show_cli_stat(detector: PromptDifferenceDetector) -> None:
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


class RunnerInteractive(RunnerInterface):
    """Represents the method for the program to interact
    with the users.

    This runner is used when the user decided to use
    the interactive mode. This is similar what Python interpreter does like."""

    def __init__(
        self,
        pipeline: PromptPipeline,
        prpt: type[TokenInterface],
        strategy: InputStrategy,
    ) -> None:
        self.pipeline = pipeline
        self.prpt = prpt
        self.input_strategy = strategy

    def write(self, content: str) -> None:
        sys.stdout.write(content)

    def run(self) -> None:
        cli_hooks.execute(cli_hooks.on_interactive)

        banner = None
        if banner is None:
            self.write(
                f"Sanipro (created by iigau) in interactive mode\n"
                f"Program was launched up at {time.asctime()}.\n"
            )
        elif banner:
            self.write("%s\n" % str(banner))

        while True:
            try:
                line = ""
                try:
                    line = self.input_strategy.input()
                    if line:
                        out = self.execute(line)
                        self.write(f"{out}\n")
                except EOFError as e:
                    break
            except ValueError as e:  # like unclosed parentheses
                logger.fatal(f"error: {e}")
            except KeyboardInterrupt:
                self.write("\nKeyboardInterrupt\n")
        self.write(f"\n")

    def execute(self, source) -> str:
        tokens_unparsed = self.pipeline.parse(str(source), self.prpt, auto_apply=True)
        tokens = str(self.pipeline)
        # self.write(f"\n")

        detector = PromptDifferenceDetector(tokens_unparsed, self.pipeline.tokens)
        show_cli_stat(detector)

        return tokens


class RunnerNonInteractive(RunnerInterface):
    """Represents the method for the program to interact
    with the users in non-interactive mode.

    Intended the case where the users feed the input from STDIN.
    """

    def __init__(
        self,
        pipeline: PromptPipeline,
        prpt: type[TokenInterface],
        strategy: InputStrategy,
    ) -> None:
        self.pipeline = pipeline
        self.prpt = prpt
        self.input_strategy = strategy

    def _run_once(self) -> None:
        sentence = None
        try:
            sentence = self.input_strategy.input().strip()
        except (KeyboardInterrupt, EOFError):
            sys.stderr.write("\n")
            sys.exit(1)
        finally:
            if sentence is not None:
                self.pipeline.parse(str(sentence), self.prpt, auto_apply=True)
                result = str(self.pipeline)
                print(result)

    def run(self) -> None:
        self._run_once()
