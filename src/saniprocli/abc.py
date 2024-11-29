from abc import ABC, abstractmethod


class RunnerInterface(ABC):
    """Represents common interface for the program to interact
    with the users."""

    @abstractmethod
    def run(self): ...


class InputStrategy(ABC):
    """Represents the method to get a user input per prompt
    in interactive mode."""

    @abstractmethod
    def input(self, prompt: str) -> str:
        """Get a user input."""
        ...
