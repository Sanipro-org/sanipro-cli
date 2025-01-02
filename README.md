# sanipro-cli

This framework provides a customizable CLI interface for sanipro.
Users can create their own `Command` class by inheriting `saniprocli.commands.CommandBase`.

There is a live example. Please refer to the `saniproclidemo/cli.py`.


```bash
poetry install
poetry run python3 src/saniproclidemo/cli.py --help
```


## pre-commit

pre-commit is a framework for building and running git hooks.
This repository employes some of pre-commit hooks. Before you the commit
your works, please run:

```sh
pre-commit run --all-files
```
