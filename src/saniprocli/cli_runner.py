import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from sanipro.logger import logger

from saniprocli import cli_hooks
from saniprocli.abc import CliRunnable, IExecuteMultiple, IExecuteSingle, InputStrategy
from saniprocli.console import ConsoleWriter


class BannerMixin(ConsoleWriter):
    def _try_banner(self) -> None:
        """Tries to show the banner if possible,

        TODO implement an option whether to show the banner or not."""
        self._ewrite(
            f"Sanipro (created by iigau) in interactive mode\n"
            f"Program was launched up at {time.asctime()}.\n"
        )


class _Runner(CliRunnable, ABC):
    """Represents the method for the program to interact
    with the users."""

    def _on_init(self) -> None:
        """The method to be called before the actual interaction."""

    def _on_exit(self) -> None:
        """The method to be called after the actual interaction."""

    @abstractmethod
    def _start_loop(self) -> None:
        """The actual start of the interaction with the user.

        This should be the specific implementation of the process
        inside the loop."""

    @abstractmethod
    def run(self) -> None: ...


class RunnerInteractive(_Runner, BannerMixin):
    """This runner is used when the user decided to use
    the interactive mode."""

    def run(self) -> None:
        cli_hooks.execute(cli_hooks.on_interactive)
        self._try_banner()
        self._on_init()
        self._start_loop()
        self._on_exit()
        self._ewrite(f"\n")


class RunnerDeclarative(_Runner):
    """Represents the method for the program to interact
    with the users in delcarative mode.

    Intended the case where the users feed the input from STDIN.
    """

    def run(self) -> None:
        cli_hooks.execute(cli_hooks.on_interactive)
        self._on_init()
        self._start_loop()
        self._on_exit()


class ExecuteSingle(ConsoleWriter, IExecuteSingle, ABC):
    """Represents the runner with the interactive user interface
    that expects a single input of the prompt."""

    _input_strategy: InputStrategy

    @abstractmethod
    def _execute_single_inner(self, source: str) -> str:
        """Implements specific features that rely on inherited class."""

    def _execute_single(self, source: str) -> str:
        return self._execute_single_inner(source)

    def _start_loop(self) -> None:
        while True:
            try:
                try:
                    prompt_input = self._input_strategy.input()
                    if prompt_input:
                        out = self._execute_single(prompt_input)
                        if out:
                            self._write(f"{out}\n")
                except EOFError:
                    break
            except Exception as e:  # like unclosed parentheses
                logger.fatal(f"error: {e}")


class _InputState(Enum):
    FIRST_INPUT = auto()
    SECOND_INPUT = auto()
    EXECUTE = auto()


@dataclass
class _InputContext:
    first: str = ""
    second: str = ""
    original_color: str = ""
    state: _InputState = _InputState.FIRST_INPUT


class ExecuteMultiple(ConsoleWriter, IExecuteMultiple, ABC):
    """Represents the runner with the interactive user interface
    that expects two different prompts."""

    from sanipro.promptset import SetCalculatorWrapper

    _input_strategy: InputStrategy
    _calculator: SetCalculatorWrapper

    @abstractmethod
    def _execute_multi_inner(self, first: str, second: str) -> str:
        """Implements specific features that rely on inherited class."""

    def _execute_multi(self, first: str, second: str) -> str:
        return self._execute_multi_inner(first, second)

    def _handle_input(self) -> str:
        while True:
            try:
                prompt_input = self._input_strategy.input()
                if prompt_input:
                    return prompt_input
            except EOFError:
                raise EOFError("EOF received. Going back to previous state.")
            except Exception as e:
                logger.fatal(f"error: {e}")

    def _handle_first_input(self) -> _InputState:
        try:
            self._ctx.first = self._handle_input()
            if self._ctx.first:
                return _InputState.SECOND_INPUT
        except EOFError:
            raise
        return _InputState.FIRST_INPUT

    def _handle_second_input(self) -> _InputState:
        self.color.color_foreground = "green"
        try:
            self._ctx.second = self._handle_input()
            if self._ctx.second:
                return _InputState.EXECUTE
        except EOFError:
            self.color.color_foreground = self._ctx.original_color
            return _InputState.FIRST_INPUT
        return _InputState.SECOND_INPUT

    def _handle_execution(self) -> None:
        out = self._execute_multi(self._ctx.first, self._ctx.second)
        if out:
            self._write(f"{out}\n")
        self.color.color_foreground = self._ctx.original_color

    def _process_state(self) -> bool:

        handlers = {
            _InputState.FIRST_INPUT: self._handle_first_input,
            _InputState.SECOND_INPUT: self._handle_second_input,
        }

        if self._ctx.state in handlers:
            self._ctx.state = handlers[self._ctx.state]()
            return True
        elif self._ctx.state == _InputState.EXECUTE:
            self._handle_execution()
            return False
        return True

    def _start_loop(self) -> None:
        from saniprocli import color

        self.color = color

        while True:
            try:
                self._ctx = _InputContext(original_color=color.color_foreground)
                while self._process_state():
                    continue
            except EOFError:
                break
