import typing


def execute(hooks: list[typing.Callable]) -> None:
    if hooks:
        for fun in hooks:
            fun()


on_init = []
on_interactive = []
