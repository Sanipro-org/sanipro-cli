import argparse
from abc import ABC, abstractmethod

from sanipro.abc import PromptPipelineInterface, TokenInterface


class InputStrategy(ABC):
    """Represents the method to get a user input per prompt
    in interactive mode."""

    @abstractmethod
    def input(self, prompt: str | None = None) -> str:
        """Get a user input."""
        ...


class RunnerInterface(ABC):
    """Represents common interface for the program to interact
    with the users."""

    @abstractmethod
    def __init__(
        self,
        pipeline: PromptPipelineInterface,
        prpt: type[TokenInterface],
        strategy: InputStrategy,
    ) -> None: ...

    @abstractmethod
    def run(self): ...


class CommandsInterface(ABC):
    """Custom subcommand implementation by user must implement
    the method of these."""

    @abstractmethod
    def get_pipeline(self) -> PromptPipelineInterface:
        """Gets user-defined pipeline."""
        ...

    @classmethod
    @abstractmethod
    def append_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Appends user-defined parser."""
        ...

    @classmethod
    @abstractmethod
    def append_subparser(cls, parser: argparse.ArgumentParser) -> None:
        """Appends user-defined subparser."""
        ...
