"""
=============================================================
 Credit Card Fraud Detection - Main Runner
=============================================================
 This script runs all 7 parts of the project in sequence.
 
 Usage: python main.py
 
 Requirements:
   - fraudTrain.csv and fraudTest.csv in the same directory
   - pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
=============================================================
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parts = [
    ("Part 1: Problem Definition",       "part1_problem_definition.py"),
    ("Part 2: Data Exploration",          "part2_data_exploration.py"),
    ("Part 3: Data Preprocessing",        "part3_data_preprocessing.py"),
    ("Part 4: Model Building",            "part4_model_building.py"),
    ("Part 5: Hyperparameter Tuning",     "part5_hyperparameter_tuning.py"),
    ("Part 6: Evaluation",                "part6_evaluation.py"),
    ("Part 7: Improvement",              "part7_improvement.py"),
]

def run_part(name, filename):
    filepath = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filename} not found!")
        return False

    print(f"\n{'#' * 60}")
    print(f"  RUNNING: {name}")
    print(f"  File: {filename}")
    print(f"{'#' * 60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, filepath],
        cwd=SCRIPT_DIR
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  [OK] {name} completed in {elapsed:.1f}s")
    else:
        print(f"\n  [ERROR] {name} failed with exit code {result.returncode}")
        return False
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("   CREDIT CARD FRAUD DETECTION PROJECT")
    print("   Running all 7 parts...")
    print("=" * 60)

    total_start = time.time()
    passed = 0
    failed = 0

    for name, filename in parts:
        success = run_part(name, filename)
        if success:
            passed += 1
        else:
            failed += 1

    total_time = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"   PROJECT COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Passed: {passed}/{len(parts)}")
    print(f"  Failed: {failed}/{len(parts)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")
