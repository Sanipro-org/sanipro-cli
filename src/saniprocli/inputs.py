import sys

from saniprocli.abc import InputStrategy
from saniprocli.color import style_for_readline
from saniprocli.console import ConsoleWriter


def input_last_break(prompt: str = "") -> str:
    """A workaround for preventing built-in input() to add '\\n'
    to the input.

    Once i thought it is posssible to handle it by sys.stdin,
    and .read(), but realized without using input(),
    it is hard to use the readline library."""
    txt = input(prompt)
    if txt != "":
        return "%s\n" % (txt,)
    return txt


class DirectInputStrategy(InputStrategy):
    """Calles sys.stdin.readline() to get a user input."""

    def input(self) -> str:
        """Preserves line breaks."""
        bufs = []
        while True:
            try:
                chunk = sys.stdin.readline()
                if chunk == "":  # EOF
                    if bufs:  # If there is any buffered input, return it
                        return "".join(bufs)
                    else:
                        raise EOFError
                bufs.append(chunk)
            except KeyboardInterrupt:
                exit(1)


class OnelineInputStrategy(InputStrategy, ConsoleWriter):
    """Represents the method to get a user input per prompt
    in interactive mode.
    It consumes just one line to get the input by a user."""

    def __init__(self, ps1: str = "") -> None:
        super().__init__()
        self.ps1 = ps1

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ps1={self.ps1})"

    def input(self) -> str:
        prompt = self.ps1
        try:
            return input_last_break(style_for_readline(prompt))
        except KeyboardInterrupt:
            self._ewrite("\nKeyboardInterrupt\n")
            return ""


class MultipleInputStrategy(InputStrategy, ConsoleWriter):
    """Represents the method to get a user input per prompt
    in interactive mode.

    It consumes multiple lines and reduce them to a string,
    and users must confirm their input by sending EOF (^D)."""

    def __init__(self, ps1: str = "", ps2: str = "") -> None:
        super().__init__()
        self.ps1 = ps1

        if ps1 != "" and ps2 == "":
            # try to have the same value with ps1
            self.ps2 = self.ps1
        else:
            self.ps2 = ps2

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ps1={self.ps1}, ps2={self.ps2})"

    def input(self) -> str:
        buffer = []
        _current_prompt = self.ps1
        _initial_prompt = _current_prompt

        while True:
            try:
                line = input_last_break(style_for_readline(_current_prompt))
                buffer.append(line)
                _current_prompt = self.ps2
            except KeyboardInterrupt:
                self._ewrite("\nKeyboardInterrupt\n")
                buffer.clear()
                # restore initial prompt
                _current_prompt = _initial_prompt
            except EOFError:
                if buffer:
                    sys.stdout.write("\n")
                    break
                else:
                    raise

        return "".join(buffer)
