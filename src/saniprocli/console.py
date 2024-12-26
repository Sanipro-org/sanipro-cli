import sys

from saniprocli.abc import ConsoleWritable


class ConsoleWriter(ConsoleWritable):
    def _write(self, text: str) -> None:
        """Writes the text to the standard output."""
        sys.stdout.write(text)

    def _ewrite(self, text: str) -> None:
        """Writes the text to the standard error output."""
        sys.stderr.write(text)
