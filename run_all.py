"""Run the whole pipeline in order, from raw data to the verification report."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = ["01_build_dataset.py", "02_explore.py", "03_model.py", "04_dominance.py",
         "05_validation.py", "06_offense_defense.py", "verify.py"]

for step in STEPS:
    print(f"\n========== {step} ==========", flush=True)
    result = subprocess.run([sys.executable, str(SRC / step)], cwd=SRC)
    if result.returncode != 0:
        sys.exit(f"{step} failed")
print("\nPipeline complete. See results/verification_report.txt")
