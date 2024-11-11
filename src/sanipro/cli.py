import argparse
import logging
import pprint
import sys
import time
from code import InteractiveConsole, InteractiveInterpreter
from collections.abc import Sequence

from . import cli_hooks, color, filters, utils
from .abc import TokenInterface
from .common import (Delimiter, FuncConfig, PromptBuilder, PromptBuilderV1,
                     PromptBuilderV2)
from .parser import TokenInteractive, TokenNonInteractive

logger_root = logging.getLogger()
logger = logging.getLogger(__name__)


class Subcommand(object):
    """the name definition for the subcommands"""

    MASK = "mask"
    RANDOM = "random"
    SORT = "sort"
    SORT_ALL = "sort-all"
    UNIQUE = "unique"

    @classmethod
    def get_set(cls) -> set:
        ok = set([val for val in cls.__dict__.keys() if val.isupper()])
        return ok


class Commands(utils.HasPrettyRepr):
    # features usable in parser_v1
    mask = False
    random = False
    sort = False
    sort_all = "lexicographical"
    unique = False

    # basic functions
    exclude = False
    input_delimiter = ","
    interactive = False
    output_delimiter = ", "
    roundup = 2
    ps1 = f"{color.default}>>>{color.RESET} "
    replace_to = r"%%%"
    subcommand = ""
    use_parser_v2 = False
    verbose = False

    # subcommands options
    reverse = False

    def get_logger_level(self) -> int:
        return logging.DEBUG if self.verbose else logging.INFO

    def debug(self) -> None:
        pprint.pprint(self, utils.debug_fp)

    @classmethod
    def prepare_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="displays extra amount of logs for debugging",
        )
        parser.add_argument(
            "-d",
            "--input-delimiter",
            default=cls.input_delimiter,
            help="specifies the delimiter for the original prompts",
        )
        parser.add_argument(
            "--output-delimiter",
            default=cls.output_delimiter,
            help="specifies the delimiter for the processed prompts",
        )
        parser.add_argument(
            "--ps1",
            default=cls.ps1,
            help="specifies the custom format for the prompts",
        )
        parser.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help="enables interactive input eternally",
        )
        parser.add_argument(
            "-r",
            "--roundup",
            default=cls.roundup,
            type=int,
            help="round up to x digits",
        )
        parser.add_argument(
            "-e",
            "--exclude",
            nargs="*",
            help="exclude words specified",
        )
        parser.add_argument(
            "--use_parser_v2",
            "-2",
            action="store_true",
            help="use parse_v2 instead of the default parse_v1",
        )

        subparsers = parser.add_subparsers(dest="subcommand")

        parser_mask = subparsers.add_parser(Subcommand.MASK)
        parser_mask.add_argument(
            "mask",
            nargs="*",
            help="mask words specified rather than removing them",
        )
        parser_mask.add_argument(
            "--replace-to",
            default=cls.replace_to,
            help="in combination with --mask, specifies the new string replaced to",
        )

        parser_random = subparsers.add_parser(Subcommand.RANDOM)
        parser_random.add_argument(
            "random",
            action="store_true",
            help="BE RANDOM!",
        )

        parser_sort = subparsers.add_parser(Subcommand.SORT)
        parser_sort.add_argument(
            "sort",
            action="store_true",
            help="reorder duplicate tokens with their strength to make them consecutive",
        )
        parser_sort.add_argument(
            "--reverse",
            action="store_true",
            help="the same as above but with reversed order",
        )

        parser_sort_all = subparsers.add_parser(Subcommand.SORT_ALL)
        parser_sort_all.add_argument(
            "sort-all",
            metavar="sort_law_name",
            default=cls.sort_all,
            const=cls.sort_all,
            nargs="?",
            choices=("lexicographical", "length", "strength"),
            help="reorder all the prompt (default: %(default)s)",
        )
        parser_sort_all.add_argument(
            "--reverse",
            action="store_true",
            help="the same as above but with reversed order",
        )

        parser_unique = subparsers.add_parser(Subcommand.UNIQUE)
        parser_unique.add_argument(
            "unique",
            action="store_true",
            help="reorder duplicate tokens with their strength to make them unique",
        )
        parser_unique.add_argument(
            "--reverse",
            action="store_true",
            help="the same as above but with reversed order",
        )

        return parser

    @property
    def get_delimiter(self) -> Delimiter:
        return Delimiter(
            self.input_delimiter,
            self.output_delimiter,
        )

    def get_builder_from(self, use_parser_v2: bool) -> PromptBuilder:
        delim = self.get_delimiter
        if not use_parser_v2:
            return delim.create_builder(PromptBuilderV1)
        return delim.create_builder(PromptBuilderV2)

    def get_builder(self) -> PromptBuilder:
        cfg = FuncConfig

        if self.use_parser_v2 and self.subcommand in Subcommand.get_set():
            raise NotImplementedError(
                f"the '{self.subcommand}' command is not available "
                "when using parse_v2."
            )

        if self.use_parser_v2:
            logger.warning("using parser_v2.")

        builder = self.get_builder_from(self.use_parser_v2)
        # always round
        builder.append_hook(
            cfg(
                func=filters.round_up,
                kwargs={"digits": self.roundup},
            ),
        )

        if self.subcommand == Subcommand.RANDOM:
            builder.append_hook(
                cfg(
                    func=filters.random,
                    kwargs={},
                ),
            )

        if self.subcommand == Subcommand.SORT_ALL:
            from . import sort_all_factory

            sorted_partial = sort_all_factory.apply_from(self.sort_all)
            builder.append_hook(
                cfg(
                    func=filters.sort_all,
                    kwargs={
                        "sorted_partial": sorted_partial,
                        "reverse": True if (self.reverse or False) else False,
                    },
                )
            )

        if self.subcommand == Subcommand.SORT:
            builder.append_hook(
                cfg(
                    func=filters.sort,
                    kwargs={
                        "reverse": (self.reverse or False),
                    },
                )
            )

        if self.subcommand == Subcommand.UNIQUE:
            builder.append_hook(
                cfg(
                    func=filters.unique,
                    kwargs={
                        "reverse": self.reverse or False,
                    },
                )
            )

        if self.subcommand == Subcommand.MASK:
            builder.append_hook(
                cfg(
                    func=filters.mask,
                    kwargs={
                        "excludes": self.mask,
                        "replace_to": self.replace_to,
                    },
                )
            )

        if self.exclude:
            builder.append_hook(
                cfg(
                    func=filters.exclude,
                    kwargs={
                        "excludes": self.exclude,
                    },
                ),
            )

        return builder

    @classmethod
    def from_sys_argv(cls, arg_val: Sequence) -> "Commands":
        parser = cls.prepare_parser()
        args = parser.parse_args(arg_val, namespace=cls())

        return args


class Runner(utils.HasPrettyRepr):
    def __init__(
        self,
        builder: PromptBuilder,
        ps1: str,
        prpt: type[TokenInterface],
    ):
        self.builder = builder
        self.ps1 = ps1
        self.prpt = prpt

    def _run_once(
        self,
    ) -> None:
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    @staticmethod
    def from_args(args: Commands) -> "Runner":
        builder = args.get_builder()
        if args.interactive:
            return RunnerInteractive(
                builder,
                ps1=args.ps1,
                prpt=TokenInteractive,
            )
        else:
            return RunnerNonInteractive(
                builder,
                ps1="",
                prpt=TokenNonInteractive,
            )


class RunnerInteractive(Runner, InteractiveConsole):
    def __init__(
        self,
        builder: PromptBuilder,
        ps1: str,
        prpt: type[TokenInterface],
    ):
        self.builder = builder
        self.ps1 = ps1
        self.prpt = prpt

        InteractiveInterpreter.__init__(self)
        self.filename = "<console>"
        self.local_exit = False
        self.resetbuffer()

    def run(self):
        cli_hooks.execute(cli_hooks.interactive)
        self.interact()

    def interact(self, banner=None, exitmsg=None):
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = self.ps1

        if banner is None:
            self.write(
                f"Sanipro (created by iigau) in interactive mode\n"
                f"Program was launched up at {time.asctime()}.\n"
            )
        elif banner:
            self.write("%s\n" % str(banner))

        try:
            while True:
                try:
                    prompt = sys.ps1
                    try:
                        line = self.raw_input(prompt)  # type: ignore
                    except EOFError:
                        break
                    else:
                        self.push(line)
                except ValueError as e:
                    logger.exception(f"error: {e}")
                except (IndexError, KeyError, AttributeError) as e:
                    logger.exception(f"error: {e}")
                except KeyboardInterrupt:
                    self.resetbuffer()
                    break

        finally:
            if exitmsg is None:
                self.write("\n")
            elif exitmsg != "":
                self.write("%s\n" % exitmsg)

    def runcode(self, code):
        print(code)

    def runsource(self, source, filename="<input>", symbol="single"):
        self.builder.parse(
            str(source),
            self.prpt,
            auto_apply=True,
        )
        result = str(self.builder)
        self.runcode(result)  # type: ignore
        return False

    def push(self, line, filename=None, _symbol="single"):
        self.buffer.append(line)
        source = "\n".join(self.buffer)
        if filename is None:
            filename = self.filename
        more = self.runsource(source, filename, symbol=_symbol)
        if not more:
            self.resetbuffer()
        return more


class RunnerNonInteractive(Runner):
    def _run_once(self) -> None:
        sentence = input(self.ps1).strip()
        if sentence != "":
            self.builder.parse(
                sentence,
                self.prpt,
                auto_apply=True,
            )
            result = str(self.builder)
            print(result)

    def run(self):
        self._run_once()


def app():
    try:
        args = Commands.from_sys_argv(sys.argv[1:])
        cli_hooks.execute(cli_hooks.init)
        logger_root.setLevel(args.get_logger_level())
        args.debug()
        runner = Runner.from_args(args)
        runner.run()
    except KeyboardInterrupt as e:
        print()
        sys.exit(1)
    except EOFError as e:
        print()
        sys.exit(1)
    except NotImplementedError as e:
        logger.error(f"error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"error: {e}")
        sys.exit(1)
