import argparse
from abc import ABC, abstractmethod

from sanipro.abc import PromptPipelineInterface, TokenInterface
from sanipro.pipeline import PromptPipeline
from sanipro.promptset import SetCalculatorWrapper


class InputStrategy(ABC):
    """Represents the method to get a user input per prompt
    in interactive mode."""

    @abstractmethod
    def input(self, prompt: str = "") -> str:
        """Get a user input."""
        ...


class CliRunnable(ABC):
    """Represents common interface for the program to interact
    with the users."""

    @abstractmethod
    def run(self):
        """Start processing."""


class CliRunnableInnerRun(ABC):
    """Contains while loop"""

    @abstractmethod
    def _start_loop(self) -> None:
        """The actual start of the interaction with the user.

        This should be the specific implementation of the process
        inside the loop."""


class CliSingular(ABC):
    """Defines the basic interface for the class
    that represents the action for single input."""

    @abstractmethod
    def __init__(
        self,
        pipeline: PromptPipelineInterface,
        prpt: type[TokenInterface],
        strategy: InputStrategy,
    ) -> None:
        """Common constructor interface that handles the
        single input."""

    @abstractmethod
    def _execute_single(self, source: str) -> str:
        """Process the input prompt, and returns the text to show it later.
        Another feature can be implements into this method. Showing statistics
        regarding the prompt, for example.
        """


class CliPlural(ABC):
    """Defines the basic interface for the class that
    represents the action for dual input."""

    @abstractmethod
    def __init__(
        self,
        pipeline: PromptPipelineInterface,
        prpt: type[TokenInterface],
        strategy: InputStrategy,
        calculator: SetCalculatorWrapper,
    ) -> None:
        """Common constructor for handling two inputs.
        The `calculator` instance operates set calculation."""

    @abstractmethod
    def _execute_multi(self, first: str, second: str) -> str:
        """Process the two prompts and return the text to show it later.
        Another feature can be implements into this method. Showing statistics
        regarding the prompt, for example.
        """


class ParserAppendable(ABC):
    """Traits that appends user-defined parser."""

    @classmethod
    @abstractmethod
    def _append_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Appends user-defined parser."""
        ...


class SubParserAppendable(ABC):
    """Traits that appends user-defined subparser."""

    @classmethod
    @abstractmethod
    def _append_subparser(cls, parser: argparse.ArgumentParser) -> None:
        """Appends user-defined subparser."""
        ...


class PipelineGettable(ABC):
    """Represents user-defined pipeline."""

    @abstractmethod
    def get_pipeline(self) -> PromptPipeline:
        """Gets user-defined pipeline."""
        ...


class CommandsInterface(ABC):
    """Custom subcommand implementation by user must implement
    the method of these."""

    ...
