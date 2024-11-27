import dataclasses
import logging
import pprint
import sys
import time
from code import InteractiveConsole, InteractiveInterpreter
from collections.abc import MutableSequence

from sanipro.abc import RunnerInterface, TokenInterface
from sanipro.common import MutablePrompt, PromptPipeline
from sanipro.filters.utils import collect_same_tokens
from sanipro.parser import TokenInteractive, TokenNonInteractive
from sanipro.utils import HasPrettyRepr

from saniprocli import cli_hooks

from .commands import CommandsBase
from .utils import get_debug_fp

logger_root = logging.getLogger()

logger = logging.getLogger(__name__)


class Runner(HasPrettyRepr, RunnerInterface):
    def __init__(
        self, pipeline: PromptPipeline, ps1: str, prpt: type[TokenInterface]
    ) -> None:
        self.pipeline = pipeline
        self.ps1 = ps1
        self.prpt = prpt

    @staticmethod
    def from_args(args: CommandsBase) -> "Runner":
        pipeline = args.get_pipeline()
        if args.interactive:
            return RunnerInteractive(pipeline, ps1=args.ps1, prpt=TokenInteractive)
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


class RunnerInteractive(Runner, InteractiveConsole):
    def __init__(
        self, pipeline: PromptPipeline, ps1: str, prpt: type[TokenInterface]
    ) -> None:
        self.pipeline = pipeline
        self.ps1 = ps1
        self.prpt = prpt

        InteractiveInterpreter.__init__(self)
        self.filename = "<console>"
        self.local_exit = False
        self.resetbuffer()

    def run(self):
        cli_hooks.execute(cli_hooks.interactive)
        self.interact()

    def interact(self, banner=None, exitmsg=None):
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = self.ps1

        if banner is None:
            self.write(
                f"Sanipro (created by iigau) in interactive mode\n"
                f"Program was launched up at {time.asctime()}.\n"
            )
        elif banner:
            self.write("%s\n" % str(banner))

        try:
            while True:
                try:
                    prompt = sys.ps1
                    try:
                        line = self.raw_input(prompt)  # type: ignore
                    except EOFError:
                        break
                    else:
                        self.push(line)
                except ValueError as e:  # like unclosed parentheses
                    logger.fatal(f"error: {e}")
                    self.resetbuffer()
                except KeyboardInterrupt:
                    self.resetbuffer()
                    self.write("\nKeyboardInterrupt\n")

        finally:
            if exitmsg is None:
                self.write("\n")
            elif exitmsg != "":
                self.write("%s\n" % exitmsg)

    def runcode(self, code):
        print(code)

    def runsource(self, source, filename="<input>", symbol="single"):
        tokens_unparsed = self.pipeline.parse(str(source), self.prpt, auto_apply=True)
        tokens = str(self.pipeline)

        anal = AnalyzerDiff(tokens_unparsed, self.pipeline.tokens)
        pprint.pprint(anal.get_stats(), get_debug_fp())

        self.runcode(tokens)  # type: ignore
        return False

    def push(self, line, filename=None, _symbol="single"):
        self.buffer.append(line)
        source = "\n".join(self.buffer)
        if filename is None:
            filename = self.filename
        more = self.runsource(source, filename, symbol=_symbol)
        if not more:
            self.resetbuffer()
        return more


class RunnerNonInteractive(Runner):
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
