"""
run_all.py
----------
Runs the full processing pipeline in order.
Execute from the project root:

    python run_all.py

Optional flags:
    python run_all.py --from 03 # start from script 03 onwards
    python run_all.py --only 02 05 # run only scripts 02 and 05
"""

import subprocess
import sys
import time
import argparse

SCRIPTS = [
   # ("01", "processing/01_clean_street_data.py"),
    ("02", "processing/02_economic_analysis.py"),
    ("03", "processing/03_deprivation_correlations.py"),
    ("04", "processing/04_train_model.py"),
    ("05", "processing/05_vulnerability_index.py"),
    ("06", "processing/06_stop_search.py"),
]


def run_script(number: str, path: str) -> bool:
    """Run a single script. Returns True on success, False on failure."""
    print(f"\n{'='*60}")
    print(f"  [{number}] {path}")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, path],
        # Don't capture output — let it stream to the terminal in real time
    )

    elapsed = round(time.time() - start, 1)

    if result.returncode == 0:
        print(f"\n  ✓ Completed in {elapsed}s")
        return True
    else:
        print(f"\n  ✗ FAILED (exit code {result.returncode}) after {elapsed}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the London Crime pipeline")
    parser.add_argument(
        "--from", dest="from_script", metavar="N",
        help="Start from script N (e.g. --from 03 skips 01 and 02)"
    )
    parser.add_argument(
        "--only", dest="only_scripts", metavar="N", nargs="+",
        help="Run only the specified script numbers (e.g. --only 02 05)"
    )
    args = parser.parse_args()

    # Determine which scripts to run
    scripts_to_run = SCRIPTS

    if args.only_scripts:
        scripts_to_run = [
            (n, p) for n, p in SCRIPTS if n in args.only_scripts
        ]
        not_found = set(args.only_scripts) - {n for n, _ in scripts_to_run}
        if not_found:
            print(f"Warning: script numbers not found: {', '.join(sorted(not_found))}")

    elif args.from_script:
        numbers = [n for n, _ in SCRIPTS]
        if args.from_script not in numbers:
            print(f"Error: script '{args.from_script}' not found. "
                  f"Valid numbers: {', '.join(numbers)}")
            sys.exit(1)
        idx = numbers.index(args.from_script)
        scripts_to_run = SCRIPTS[idx:]

    if not scripts_to_run:
        print("No scripts to run.")
        sys.exit(0)

    # Run
    overall_start = time.time()
    results = {}

    for number, path in scripts_to_run:
        success = run_script(number, path)
        results[number] = success
        if not success:
            print(f"\nPipeline stopped at script {number}.")
            print("Fix the error above and rerun with:  python run_all.py --from {number}")
            break

    # Summary
    total = round(time.time() - overall_start, 1)
    passed = sum(results.values())
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"  Pipeline summary  ({total}s total)")
    print(f"{'='*60}")
    for number, path in scripts_to_run:
        if number in results:
            icon = "✓" if results[number] else "✗"
            print(f"  {icon} [{number}] {path}")
        else:
            print(f"  - [{number}] {path}  (skipped)")

    print()
    if failed == 0:
        print(f"  All {passed} scripts passed.")
        print("\n  Start the dashboard with:  streamlit run app.py")
    else:
        print(f"  {passed} passed, {failed} failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()