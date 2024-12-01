import functools

import click

color_foreground = "cyan"

style = functools.partial(click.style, fg=color_foreground)


def style_for_readline(text: str) -> str:
    """Styles a text with ANSI style and readline compatibility,
    and returns the new string."""

    styled = style(text)
    if text == styled:
        return text
    return f"\001{styled}\002"
