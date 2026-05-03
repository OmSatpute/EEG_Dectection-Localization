import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


def run_step(name, command, cwd):
    print(f"\n=== {name} ===")
    subprocess.run(command, cwd=cwd, check=True)


def main():
    python = sys.executable

    run_step(
        "SVM detection",
        [python, str(ROOT_DIR / "src" / "svm.py")],
        DATA_DIR,
    )

    run_step(
        "CNN detection",
        [python, "-m", "src.train_cnn"],
        ROOT_DIR,
    )

    run_step(
        "Grad-CAM localization",
        [python, "-m", "src.localization"],
        ROOT_DIR,
    )

    print("\nDetection and localization workflow complete.")


if __name__ == "__main__":
    main()
