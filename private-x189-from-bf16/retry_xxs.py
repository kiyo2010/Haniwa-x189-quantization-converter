"""Select the portable calibration helper explicitly in the managed XXS subprocess."""
import importlib.util
from pathlib import Path
import sys

import rebuild


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]/"private-x189-github-prep"
    load_local("calibrate", base/"calibrate.py")
    load_local("x189_recipe", base/"x189_recipe.py")
    rebuild.main()
