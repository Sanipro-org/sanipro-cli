import logging
from typing import Generator, Type

from .abc import TokenInterface

logger = logging.getLogger()

class Token(TokenInterface):
    def __init__(self, name: str, strength: float) -> None:
        self._name = name
        self._strength = float(strength)
        self._delimiter = None

    @property
    def name(self):
        return self._name

    @property
    def strength(self):
        return self._strength

    @property
    def length(self):
        return len(self.name)

    def replace(self, replace: str):
        return type(self)(replace, self._strength)

    def __repr__(self):
        items = (f"{v!r}" for v in (self.name, self.strength))
        return "{}({})".format(type(self).__name__, Tokens.COMMA.join(items))


class TokenInteractive(Token):
    def __init__(self, name: str, strength: float):
        Token.__init__(self, name, strength)
        self._delimiter = ":"

    def __str__(self):
        if self.strength != 1.0:
            return "({}{}{:.1f})".format(self.name, self._delimiter, self.strength)
        return self.name


class TokenNonInteractive(Token):
    def __init__(self, name: str, strength: float):
        Token.__init__(self, name, strength)
        self._delimiter = "\t"

    def __str__(self):
        return "{}{}{.2f}".format(self.strength, self._delimiter, self.name)


class Tokens:
    PARENSIS_LEFT = "("
    PARENSIS_RIGHT = ")"
    COLON = ":"
    COMMA = ","
    SPACE = " "


def extract_token(sentence: str, delimiter: str) -> list[str]:
    """
    split `sentence` at commas and remove parentheses.

    >>> list(extract_token('1girl,'))
    ['1girl']
    >>> list(extract_token('(brown hair:1.2),'))
    ['brown hair:1.2']
    >>> list(extract_token('1girl, (brown hair:1.2), school uniform, smile,'))
    ['1girl', 'brown hair:1.2', 'school uniform', 'smile']
    """

    product = []
    parenthesis = []
    character_stack = []

    for index, character in enumerate(sentence):
        if character == Tokens.PARENSIS_LEFT:
            parenthesis.append(index)
        elif character == Tokens.PARENSIS_RIGHT:
            parenthesis.pop()
        elif character == delimiter:
            if parenthesis:
                character_stack.append(character)
                continue
            element = "".join(character_stack).strip()
            character_stack.clear()
            product.append(element)
        else:
            character_stack.append(character)

    if parenthesis:
        first_parenthesis_index = parenthesis[0]
        raise ValueError(f"first unclosed parenthesis was found after {sentence[0:first_parenthesis_index]!r}")

    return product

def parse_line(
    token_combined: str, token_factory: Type[TokenInterface]
) -> TokenInterface:
    """
    split `token_combined` into left and right sides with `:`
    when there are three or more elements,
    the right side separated by the last colon is adopted as the strength.

    >>> from lib.common import PromptInteractive, PromptNonInteractive

    >>> parse_line('brown hair:1.2', PromptInteractive)
    PromptInteractive(_name='brown hair', _strength='1.2', _delimiter=':')

    >>> parse_line('1girl', PromptInteractive)
    PromptInteractive(_name='1girl', _strength='1.0', _delimiter=':')

    >>> parse_line('brown:hair:1.2', PromptInteractive)
    PromptInteractive(_name='brown:hair', _strength='1.2', _delimiter=':')
    """
    token = token_combined.split(Tokens.COLON)
    token_length = len(token)

    match (token_length):
        case 1:
            name, *_ = token
            return token_factory(name, 1.0)
        case 2:
            name, strength, *_ = token
            return token_factory(name, float(strength))
        case _:
            *ret, strength = token
            name = Tokens.COLON.join(ret)
            return token_factory(name, float(strength))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
