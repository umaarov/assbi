"""Project entry point.

Running this file (e.g. the green ▶ button in PyCharm) executes the
dependency-free synthetic demo: it runs the full surveillance pipeline,
populates the analytics warehouse, prints a BI report and shows the AI
assistant answering questions.

For the full CLI use:  python -m assbi.cli --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from assbi import cli  # noqa: E402


def main() -> None:
    cli._force_utf8()
    print("ASSBI — running the dependency-free synthetic demo.")
    print("(For the full CLI: python -m assbi.cli --help)\n")
    # Equivalent to: assbi run --session demo
    raise SystemExit(cli.main(["run", "--session", "demo"]))


if __name__ == "__main__":
    main()
