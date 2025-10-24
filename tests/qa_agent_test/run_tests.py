"""
Test runner for the QA agent module tests.

This script runs all tests for the QA agent module and provides
a summary of the results.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


def run_qa_agent_tests():
    """Run all QA agent module tests."""
    print("🧪 Running QA Agent Module Tests...")
    print("=" * 50)
    
    # Test directory
    test_dir = Path(__file__).parent
    
    # Run tests with coverage
    args = [
        str(test_dir),
        "-v",
        "--tb=short",
        "--cov=qa_agent",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/qa_agent",
        "--cov-fail-under=80"
    ]
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✅ All QA agent tests passed!")
        print("📊 Coverage report generated in htmlcov/qa_agent/")
    else:
        print(f"\n❌ Some tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_qa_agent_tests()
    sys.exit(exit_code)
