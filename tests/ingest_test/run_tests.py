#!/usr/bin/env python3
"""
Test runner for the ingestion module tests.

This script provides a convenient way to run all ingestion module tests
with different configurations and reporting options.
"""

import sys
import pytest
from pathlib import Path

# Add the project root to Python path to ensure imports work
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_tests(verbose=False, coverage=False, specific_test=None):
    """
    Run the ingestion module tests.
    
    Args:
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        specific_test: Run only a specific test file
    """
    # Get the test directory
    test_dir = Path(__file__).parent
    
    # Build pytest arguments
    args = [str(test_dir)]
    
    if verbose:
        args.append("-v")
    
    if coverage:
        args.extend([
            "--cov=ingest",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    if specific_test:
        args.append(f"test_{specific_test}.py")
    
    # Add common options
    args.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    # Run tests
    exit_code = pytest.main(args)
    return exit_code


def main():
    """Main entry point for the test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ingestion module tests")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("-c", "--coverage", action="store_true",
                       help="Enable coverage reporting")
    parser.add_argument("-t", "--test", type=str,
                       help="Run specific test file (without test_ prefix and .py extension)")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests with verbose output and coverage")
    
    args = parser.parse_args()
    
    if args.all:
        verbose = True
        coverage = True
        specific_test = None
    else:
        verbose = args.verbose
        coverage = args.coverage
        specific_test = args.test
    
    print("🧪 Running LERK System Ingestion Module Tests")
    print("=" * 50)
    
    if specific_test:
        print(f"Running specific test: {specific_test}")
    else:
        print("Running all tests")
    
    if coverage:
        print("Coverage reporting enabled")
    
    if verbose:
        print("Verbose output enabled")
    
    print()
    
    # Run tests
    exit_code = run_tests(verbose, coverage, specific_test)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
