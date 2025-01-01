import functools
import re

import click

color_foreground = "cyan"

style = functools.partial(click.style, fg=color_foreground)


def style_for_readline(text: str) -> str:
    """Styles a text with ANSI style and readline compatibility,
    and returns the new string."""
    re_escaped = re.compile(r"(\033\[\d+m)")
    _style = functools.partial(click.style, fg=color_foreground)

    _text = _style(text)
    if text == _text:
        return text

    _text = re_escaped.sub(r"\001\1\002", _text)
    return _text
