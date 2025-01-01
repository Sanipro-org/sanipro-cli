import logging
import typing

from sanipro.logger import logger


class BufferingLoggerWriter(typing.IO):
    """Pseudo file object redirected to logger."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message) -> int:
        if "\n" not in message:
            self.buffer += message
        else:
            parts = message.split("\n")
            if self.buffer:
                s = self.buffer + parts.pop(0)
                self.logger.log(self.level, s)
            self.buffer = parts.pop()
            for part in parts:
                self.logger.log(self.level, part)
        return 0

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# Mainly for `pprint.pprint()`
logger_fp = BufferingLoggerWriter(logger, logging.DEBUG)


def get_log_level_from(count: int | None) -> int:
    """Map function that maps the number of options to log level."""

    if count is not None:
        if count > 1:
            return logging.DEBUG
    return logging.INFO
