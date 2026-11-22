# SPDX-License-Identifier: Apache-2.0
"""Convert BF16 language models to standard GGUF with Haniwa X189."""
import argparse
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="python -m x189_clean",
        description=__doc__,
        epilog="Use convert --help for conversion options, list-models to list model families, "
               "or gui to open the desktop converter.",
    )
    parser.add_argument("command", nargs="?", choices=("convert", "list-models", "gui"))
    command = parser.parse_args(args[:1]).command
    if command is None:
        parser.print_help()
        return
    if command == "gui":
        from .gui import main as gui_main

        return gui_main(args[1:])
    from .generic import cli

    return cli(command, args[1:])


if __name__ == "__main__":
    sys.exit(main())
