"""Run the complete offline course with a non-interactive plotting backend."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    ("curve_fitting", "--output", "curve_fitting"),
    ("interlab_statistics", "--output", "statistics"),
    ("dynamics", "--out", "dynamics"),
    ("differential_equations", "--outdir", "13-differential-equations"),
    ("monte_carlo", "--output", "monte_carlo"),
    ("wavelets", "--output", "wavelets"),
    ("optimization", "--output", "optimization"),
    ("genetic_algorithm", "--output", "genetic_algorithm"),
    ("queueing", "--output", "queueing"),
    ("game_theory", "--output", "game-theory"),
    ("ahp", "--output", "ahp"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    for script, flag, folder in EXAMPLES:
        print(f"\n--- {script} ---", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "examples" / f"{script}.py"), flag, str(output / folder)],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    print(f"\nCompleted {len(EXAMPLES)} examples. Output: {output}")


if __name__ == "__main__":
    main()
