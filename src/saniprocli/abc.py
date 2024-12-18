import argparse
from abc import ABC, abstractmethod

from sanipro.abc import MutablePrompt, Prompt, PromptPipelineInterface, TokenInterface
from sanipro.pipeline import PromptPipeline


class InputStrategy(ABC):
    """Represents the method to get a user input per prompt
    in interactive mode."""

    @abstractmethod
    def input(self, prompt: str = "") -> str:
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


class CommandExecutable(ABC):
    @abstractmethod
    def execute(self, prompt: Prompt) -> MutablePrompt: ...


class ParserAppendable:
    @abstractmethod
    def _append_parser(self, parser: argparse.ArgumentParser) -> None:
        """Appends user-defined parser."""
        ...


class SubParserAppendable:
    @abstractmethod
    def _append_subparser(self, parser: argparse.ArgumentParser) -> None:
        """Appends user-defined subparser."""
        ...


class CliArgsNamespace(ParserAppendable, SubParserAppendable): ...


class PipelineGettable:
    @abstractmethod
    def get_pipeline(self) -> PromptPipeline:
        """Gets user-defined pipeline."""
        ...


class CommandsInterface(ABC):
    """Custom subcommand implementation by user must implement
    the method of these."""

    ...
