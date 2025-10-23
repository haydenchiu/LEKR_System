"""
Test runner for the logic_extractor module.

This script runs all tests for the logic_extractor module and provides
detailed test results and coverage information.
"""

import sys
import os
import pytest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run all logic_extractor tests with coverage."""
    
    # Test directory
    test_dir = Path(__file__).parent
    
    # Coverage configuration
    coverage_args = [
        "--cov=logic_extractor",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80"
    ]
    
    # Test arguments
    test_args = [
        str(test_dir),
        "-v",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings",
        "--color=yes"
    ]
    
    # Combine all arguments
    all_args = test_args + coverage_args
    
    print("🧪 Running Logic Extractor Tests")
    print("=" * 50)
    print(f"Test directory: {test_dir}")
    print(f"Coverage target: 80%")
    print("=" * 50)
    
    # Run tests
    exit_code = pytest.main(all_args)
    
    if exit_code == 0:
        print("\n🎉 All tests passed!")
        print("📊 Coverage report generated in htmlcov/")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
