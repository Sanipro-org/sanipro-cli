import functools
import re

import click

color_foreground = "cyan"

style = functools.partial(click.style, fg=color_foreground)


class EscSeqWrapper:
    esc_sequence_regex = r"(\033\[\d+(?:;\d+)*m)"
    pattern = re.compile(esc_sequence_regex)

    @staticmethod
    def wrap(text: str) -> str:
        """Wrap ANSI escape sequences between \1 and \2 for GNU readline."""

        return EscSeqWrapper.pattern.sub(r"\01\g<1>\02", text)


def style_for_readline(text: str) -> str:
    """Styles a text with ANSI style and readline compatibility,
    and returns the new string."""

    style_partial = functools.partial(click.style, fg=color_foreground)
    text_styled = style_partial(text)

    if text == text_styled:
        return text

    return EscSeqWrapper.wrap(text_styled)
