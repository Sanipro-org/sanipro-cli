import os.path
import subprocess


def main():
    target = os.path.join(os.path.dirname(__file__), "../../..")

    cmds = [
        ["pipx", "run", "isort", "-m3", target],
        ["pipx", "run", "black", "-C", target],
        ["pipx", "run", "autoflake", "--remove-all-unused-imports", "-ir", target],
    ]

    for cmd in cmds:
        subprocess.run(cmd, capture_output=False)
