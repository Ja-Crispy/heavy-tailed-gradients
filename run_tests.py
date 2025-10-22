"""
Comprehensive test runner for Phase 1 implementation.

Runs all tests and provides detailed validation report.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'─'*60}")
    print(f"Running: {description}")
    print(f"{'─'*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(result.stdout)

        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True
        else:
            print(f"❌ {description} FAILED")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️  {description} TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ {description} ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PHASE 1 COMPREHENSIVE TEST SUITE")
    print("="*60)

    results = {}

    # Change to tail-scaling directory
    tail_scaling_dir = Path(__file__).parent
    import os
    os.chdir(tail_scaling_dir)

    print(f"\nWorking directory: {Path.cwd()}")

    # Test 1: Tail estimator unit tests
    results['Tail Estimator Tests'] = run_command(
        f"{sys.executable} tests/test_tail_estimators.py",
        "Tail Estimator Unit Tests"
    )

    # Test 2: Smoke tests (includes model tests, data tests, and mini experiments)
    results['Smoke Tests'] = run_command(
        f"{sys.executable} tests/smoke_test.py",
        "End-to-End Smoke Tests"
    )

    # Test 3: Model self-tests
    results['MinimalFFN Tests'] = run_command(
        f"{sys.executable} models/minimal_ffn.py",
        "MinimalFFN Self Tests"
    )

    results['NanoTransformer Tests'] = run_command(
        f"{sys.executable} models/nano_transformer.py",
        "NanoTransformer Self Tests"
    )

    # Test 4: Data generation tests
    results['Synthetic Data Tests'] = run_command(
        f"{sys.executable} experiments/synthetic_data.py",
        "Synthetic Data Generation Tests"
    )

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:40s} {status}")

    print("\n" + "="*60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("="*60)

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour Phase 1 implementation is validated and ready for experiments!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. (Optional) Configure wandb: wandb login")
        print("  3. Run Experiment 1.1: python experiments/measure_alpha.py --config config/experiment_1_1.yaml")
        print("  4. Run Experiment 1.2: python experiments/measure_alpha.py --config config/experiment_1_2.yaml")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease review the errors above and fix the failing tests.")
        print("Once all tests pass, you can proceed with experiments.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
