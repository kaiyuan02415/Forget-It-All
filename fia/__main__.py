"""Allow ``python -m fia ...`` to invoke the same CLI as ``python main.py``."""

from pathlib import Path
import runpy
import sys


def main():
    main_py = Path(__file__).resolve().parent.parent / "main.py"
    sys.argv[0] = str(main_py)
    runpy.run_path(str(main_py), run_name="__main__")


if __name__ == "__main__":
    main()
