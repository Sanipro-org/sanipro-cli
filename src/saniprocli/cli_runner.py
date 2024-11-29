import dataclasses
import logging
import pprint
import sys
import time
from collections.abc import MutableSequence

from sanipro.abc import RunnerInterface, TokenInterface
from sanipro.common import MutablePrompt, PromptPipeline
from sanipro.filters.utils import collect_same_tokens
from sanipro.parser import TokenInteractive, TokenNonInteractive
from sanipro.utils import HasPrettyRepr

from saniprocli import cli_hooks, color
from saniprocli.abc import InputStrategy

from .commands import CommandsBase
from .utils import get_debug_fp

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class OnelineInputStrategy(InputStrategy):
    """Represents the method to get a user input per prompt
    in interactive mode.
    It consumes just one line to get the input by a user."""

    def input(self, prompt: str) -> str:
        return input(prompt)


class MultipleInputStrategy(InputStrategy):
    """Represents the method to get a user input per prompt
    in interactive mode.

    It consumes multiple lines and reduce them to a string,
    and users must confirm their input by sending EOF (^D)."""

    def __init__(self):
        super().__init__()
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = f"\001{color.CYAN}...{color.RESET}\002 "

    def input(self, prompt: str) -> str:
        buffer = []
        _prompt = prompt
        while True:
            try:
                line = input(_prompt)
                buffer.append(line)
                _prompt = sys.ps2
            except EOFError:
                if buffer:
                    break
                else:
                    raise

        return "".join(buffer)


class Runner(HasPrettyRepr, RunnerInterface):
    """Represents the common method for the program to interact
    with the users."""

    def __init__(
        self, pipeline: PromptPipeline, ps1: str, prpt: type[TokenInterface]
    ) -> None:
        self.pipeline = pipeline
        self.ps1 = ps1
        self.prpt = prpt

    @staticmethod
    def from_args(
        args: CommandsBase,
        /,
        input_strategy: type[InputStrategy] = MultipleInputStrategy,
    ) -> "Runner":
        pipeline = args.get_pipeline()
        if args.interactive:
            return RunnerInteractive(
                pipeline,
                ps1=args.ps1,
                prpt=TokenInteractive,
                input_strategy=input_strategy(),
            )
        else:
            return RunnerNonInteractive(pipeline, ps1="", prpt=TokenNonInteractive)


class Analyzer:
    pass


@dataclasses.dataclass
class DiffStatistics:
    before_num: int
    after_num: int
    reduced_num: int
    duplicated_tokens: MutableSequence[MutablePrompt]


@dataclasses.dataclass
class AnalyzerDiff(Analyzer):
    before_process: MutablePrompt
    after_process: MutablePrompt

    @property
    def len_reduced(self) -> int:
        return len(self.before_process) - len(self.after_process)

    def get_duplicates(self) -> list[MutablePrompt]:
        threshould = 1
        dups = collect_same_tokens(self.before_process)
        tokens = [tokens for tokens in dups.values() if len(tokens) > threshould]
        return tokens

    def get_stats(self) -> DiffStatistics:
        stats = DiffStatistics(
            len(self.before_process),
            len(self.after_process),
            self.len_reduced,
            self.get_duplicates(),
        )
        return stats


class RunnerInteractive(Runner):
    """Represents the method for the program to interact
    with the users.

    This runner is used when the user decided to use
    the interactive mode. This is similar what Python interpreter does like."""

    def __init__(
        self,
        pipeline: PromptPipeline,
        ps1: str,
        prpt: type[TokenInterface],
        input_strategy: InputStrategy,
    ) -> None:
        self.pipeline = pipeline
        self.ps1 = ps1
        self.prpt = prpt
        self.input_strategy = input_strategy

    def write(self, content: str):
        sys.stdout.write(content)

    def run(self):
        cli_hooks.execute(cli_hooks.interactive)

        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = self.ps1

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
                prompt = sys.ps1
                try:
                    line = self.input_strategy.input(prompt)
                    if line:
                        out = self.execute(line)
                        self.write(f"{out}\n")
                    else:
                        self.write(f"\n")
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
        self.write(f"\n")

        anal = AnalyzerDiff(tokens_unparsed, self.pipeline.tokens)
        pprint.pprint(anal.get_stats(), get_debug_fp())

        return tokens


class RunnerNonInteractive(Runner):
    """Represents the method for the program to interact
    with the users in non-interactive mode.

    Intended the case where the users feed the input from STDIN.
    """

    def _run_once(self) -> None:
        sentence = None
        try:
            sentence = input(self.ps1).strip()
        except (KeyboardInterrupt, EOFError):
            sys.stderr.write("\n")
            sys.exit(1)
        finally:
            if sentence is not None:
                self.pipeline.parse(str(sentence), self.prpt, auto_apply=True)
                result = str(self.pipeline)
                print(result)

    def run(self):
        self._run_once()
