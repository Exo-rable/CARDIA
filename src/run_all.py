"""Run the whole pipeline in order:  python src/run_all.py

clean -> EDA -> train -> explain -> app smoke test.  Each stage can also be run on its own.
Add --skip-test to skip the app test, --skip-train to reuse the existing saved model.
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

STAGES = [
    ("Clean and validate data", SRC / "02_clean.py"),
    ("Exploratory analysis", SRC / "01_eda.py"),
    ("Train, tune and evaluate", SRC / "03_train.py"),
    ("Explain the model", SRC / "04_explain.py"),
    ("Test the app", ROOT / "tests" / "smoke_test.py"),
]


def main():
    skip_test = "--skip-test" in sys.argv
    skip_train = "--skip-train" in sys.argv
    for name, script in STAGES:
        if skip_test and script.name == "smoke_test.py":
            continue
        if skip_train and script.name == "03_train.py":
            continue
        print(f"\n=== {name} ({script.name}) ===", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"Stage failed: {name}. Fix the error above and rerun.")
        print(f"--- done in {time.time() - t0:.0f}s", flush=True)
    print("\nAll stages finished. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
