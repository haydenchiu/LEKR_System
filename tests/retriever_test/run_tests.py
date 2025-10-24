#!/usr/bin/env python3
"""
Test runner for the retriever module.

This script runs all tests for the retriever module and provides
detailed coverage reporting and test results.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_retriever_tests():
    """Run all retriever module tests with coverage."""
    print("=" * 60)
    print("LERK Retriever Module Test Suite")
    print("=" * 60)
    
    # Test directory
    test_dir = Path(__file__).parent
    
    # Run tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--cov=retriever",  # Coverage for retriever module
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html:htmlcov/retriever",  # HTML coverage report
        "--cov-fail-under=80",  # Fail if coverage < 80%
        "-x",  # Stop on first failure
        "--disable-warnings"  # Disable warnings for cleaner output
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n" + "=" * 60)
        print("✅ All retriever tests passed!")
        print("📊 Coverage report generated in htmlcov/retriever/")
        print("=" * 60)
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("❌ Some retriever tests failed!")
        print(f"Exit code: {e.returncode}")
        print("=" * 60)
        return False
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return False

def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"Running specific test: {test_file}")
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print(f"✅ Test {test_file} passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test {test_file} failed!")
        return False

def main():
    """Main test runner function."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            sys.exit(1)
        
        success = run_specific_test(test_file)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = run_retriever_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
