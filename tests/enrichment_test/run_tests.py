#!/usr/bin/env python3
"""
Test runner for enrichment module tests.

This script runs all unit tests for the enrichment module
and provides comprehensive test reporting.
"""

import sys
import pytest
from pathlib import Path

# Add the project root to Python path to ensure imports work
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_tests():
    """Run all enrichment module tests."""
    print("LERK System - Enrichment Module Tests")
    print("=" * 50)
    
    # Test directory
    test_dir = Path(__file__).parent
    
    # Run tests with coverage
    try:
        # Run tests with coverage
        result = pytest.main([
            str(test_dir),
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--cov=enrichment",  # Coverage for enrichment module
            "--cov-report=html",  # HTML coverage report
            "--cov-report=term-missing",  # Terminal coverage report
            "--cov-fail-under=80",  # Fail if coverage below 80%
            "-x",  # Stop on first failure
            "--maxfail=5",  # Maximum failures before stopping
        ])
        
        if result == 0:
            print("\n✅ All tests passed!")
            print("📊 Coverage report generated in htmlcov/")
        else:
            print(f"\n❌ Tests failed with exit code: {result}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"Running specific test: {test_file}")
    
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return 1
    
    try:
        result = pytest.main([
            str(test_path),
            "-v",
            "--tb=short",
            "--cov=enrichment",
            "--cov-report=term-missing",
        ])
        
        return result
        
    except Exception as e:
        print(f"❌ Error running specific test: {e}")
        return 1

def run_tests_without_coverage():
    """Run tests without coverage reporting."""
    print("Running tests without coverage...")
    
    test_dir = Path(__file__).parent
    
    try:
        result = pytest.main([
            str(test_dir),
            "-v",
            "--tb=short",
            "-x",
            "--maxfail=5",
        ])
        
        return result
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def run_tests_parallel():
    """Run tests in parallel for faster execution."""
    print("Running tests in parallel...")
    
    test_dir = Path(__file__).parent
    
    try:
        result = pytest.main([
            str(test_dir),
            "-v",
            "--tb=short",
            "-n", "auto",  # Use all available CPU cores
            "--cov=enrichment",
            "--cov-report=term-missing",
        ])
        
        return result
        
    except Exception as e:
        print(f"❌ Error running parallel tests: {e}")
        return 1

def main():
    """Main function to run tests based on command line arguments."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-cov":
            return run_tests_without_coverage()
        elif sys.argv[1] == "--parallel":
            return run_tests_parallel()
        elif sys.argv[1].endswith(".py"):
            return run_specific_test(sys.argv[1])
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_tests.py                    # Run all tests with coverage")
            print("  python run_tests.py --no-cov          # Run tests without coverage")
            print("  python run_tests.py --parallel        # Run tests in parallel")
            print("  python run_tests.py test_file.py      # Run specific test file")
            print("  python run_tests.py --help            # Show this help")
            return 0
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            return 1
    else:
        return run_tests()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
