#!/usr/bin/env python3
"""
Test runner for the clustering module.

This script runs all tests for the clustering module and generates
a coverage report to ensure comprehensive testing.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run clustering module tests with coverage."""
    print("🧪 Running Clustering Tests")
    print("=" * 50)
    
    # Get the test directory
    test_dir = Path(__file__).parent
    print(f"Test directory: {test_dir}")
    print(f"Coverage target: 80%")
    print("=" * 50)
    
    # Run pytest with coverage
    cmd = [
        "python", "-m", "pytest",
        str(test_dir),
        "--cov=clustering",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "-v",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        print("🎉 All tests passed!")
        print("📊 Coverage report generated in htmlcov/")
        
    except subprocess.CalledProcessError as e:
        print("❌ Tests failed with exit code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest pytest-cov")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
