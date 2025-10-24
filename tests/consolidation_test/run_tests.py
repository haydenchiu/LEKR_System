#!/usr/bin/env python3
"""
Test runner for the consolidation module.

This script runs all tests for the consolidation module and provides
a summary of test results and coverage information.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all consolidation module tests."""
    print("Running consolidation module tests...")
    print("=" * 50)
    
    # Get the test directory
    test_dir = Path(__file__).parent
    
    try:
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v",
            "--tb=short",
            "--cov=consolidation",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov_consolidation",
            "--cov-fail-under=80"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ All tests passed successfully!")
            print("✅ Code coverage meets the 80% threshold!")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed or coverage is below 80%")
            print(f"Exit code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_specific_test_file(test_file):
    """Run tests for a specific test file."""
    print(f"Running tests for {test_file}...")
    print("=" * 50)
    
    test_path = Path(__file__).parent / test_file
    
    try:
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests for {test_file}: {e}")
        return False

def main():
    """Main function to run tests."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        success = run_specific_test_file(test_file)
    else:
        # Run all tests
        success = run_tests()
    
    if success:
        print("\n🎉 Test execution completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
