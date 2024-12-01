import typing

Callback = typing.Callable
Callbacks = list[Callback]


def execute(hooks: Callbacks) -> None:
    if hooks:
        for fun in hooks:
            fun()


on_init: Callbacks = []
on_interactive: Callbacks = []
