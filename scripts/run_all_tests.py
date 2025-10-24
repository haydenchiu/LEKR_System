#!/usr/bin/env python3
"""
LERK System - Test Runner Script
This script runs all tests for the LERK System with comprehensive reporting.
"""

import argparse
import logging
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Runs comprehensive tests for the LERK System."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the test runner.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.test_modules = [
            'ingest_test',
            'enrichment_test', 
            'logic_extractor_test',
            'clustering_test',
            'consolidation_test',
            'retriever_test',
            'qa_agent_test'
        ]
        
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_module_tests(self, module: str, verbose: bool = False, coverage: bool = False) -> Dict[str, Any]:
        """Run tests for a specific module."""
        try:
            logger.info(f"Running tests for {module}")
            
            test_dir = self.project_root / 'tests' / module
            if not test_dir.exists():
                return {
                    'success': False,
                    'error': f"Test directory not found: {test_dir}",
                    'tests_run': 0,
                    'tests_passed': 0,
                    'tests_failed': 0,
                    'duration': 0
                }
            
            # Build pytest command
            cmd = ['python', '-m', 'pytest', str(test_dir)]
            
            if verbose:
                cmd.append('-v')
            
            if coverage:
                cmd.extend(['--cov', module.replace('_test', ''), '--cov-report=html', '--cov-report=term'])
            
            # Add test discovery
            cmd.extend(['--tb=short', '--strict-markers'])
            
            # Run tests
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Parse results
            if result.returncode == 0:
                # Parse pytest output for test counts
                output_lines = result.stdout.split('\n')
                tests_run = 0
                tests_passed = 0
                tests_failed = 0
                
                for line in output_lines:
                    if 'passed' in line and 'failed' in line:
                        # Extract test counts
                        parts = line.split()
                        for part in parts:
                            if part.isdigit():
                                if 'passed' in line:
                                    tests_passed = int(part)
                                elif 'failed' in line:
                                    tests_failed = int(part)
                        tests_run = tests_passed + tests_failed
                        break
                
                return {
                    'success': True,
                    'tests_run': tests_run,
                    'tests_passed': tests_passed,
                    'tests_failed': tests_failed,
                    'duration': duration,
                    'output': result.stdout,
                    'error_output': result.stderr
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'tests_run': 0,
                    'tests_passed': 0,
                    'tests_failed': 0,
                    'duration': duration,
                    'output': result.stdout
                }
                
        except Exception as e:
            logger.error(f"Failed to run tests for {module}: {e}")
            return {
                'success': False,
                'error': str(e),
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'duration': 0
            }
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        try:
            logger.info("Running integration tests")
            
            # Look for integration test files
            integration_tests = []
            for test_dir in self.project_root.glob('tests/*_test'):
                for test_file in test_dir.glob('test_*integration*.py'):
                    integration_tests.append(str(test_file))
            
            if not integration_tests:
                return {
                    'success': True,
                    'message': 'No integration tests found',
                    'tests_run': 0,
                    'tests_passed': 0,
                    'tests_failed': 0,
                    'duration': 0
                }
            
            # Run integration tests
            cmd = ['python', '-m', 'pytest'] + integration_tests
            
            if verbose:
                cmd.append('-v')
            
            cmd.extend(['--tb=short', '--strict-markers'])
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'tests_run': 1,  # Simplified for integration tests
                    'tests_passed': 1,
                    'tests_failed': 0,
                    'duration': duration,
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'tests_run': 1,
                    'tests_passed': 0,
                    'tests_failed': 1,
                    'duration': duration,
                    'output': result.stdout
                }
                
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'duration': 0
            }
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        try:
            logger.info("Running performance tests")
            
            # Look for performance test files
            performance_tests = []
            for test_dir in self.project_root.glob('tests/*_test'):
                for test_file in test_dir.glob('test_*performance*.py'):
                    performance_tests.append(str(test_file))
            
            if not performance_tests:
                return {
                    'success': True,
                    'message': 'No performance tests found',
                    'tests_run': 0,
                    'tests_passed': 0,
                    'tests_failed': 0,
                    'duration': 0
                }
            
            # Run performance tests
            cmd = ['python', '-m', 'pytest'] + performance_tests
            
            if verbose:
                cmd.append('-v')
            
            cmd.extend(['--tb=short', '--strict-markers'])
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'tests_run': 1,  # Simplified for performance tests
                    'tests_passed': 1,
                    'tests_failed': 0,
                    'duration': duration,
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'tests_run': 1,
                    'tests_passed': 0,
                    'tests_failed': 1,
                    'duration': duration,
                    'output': result.stdout
                }
                
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'duration': 0
            }
    
    def run_all_tests(self, 
                     modules: Optional[List[str]] = None,
                     verbose: bool = False,
                     coverage: bool = False,
                     integration: bool = False,
                     performance: bool = False) -> Dict[str, Any]:
        """Run all tests."""
        self.start_time = time.time()
        
        try:
            logger.info("Starting comprehensive test run")
            
            # Determine which modules to test
            test_modules = modules or self.test_modules
            
            # Run module tests
            module_results = {}
            for module in test_modules:
                if module in self.test_modules:
                    module_results[module] = self.run_module_tests(module, verbose, coverage)
                else:
                    module_results[module] = {
                        'success': False,
                        'error': f"Unknown module: {module}",
                        'tests_run': 0,
                        'tests_passed': 0,
                        'tests_failed': 0,
                        'duration': 0
                    }
            
            # Run integration tests if requested
            integration_results = None
            if integration:
                integration_results = self.run_integration_tests(verbose)
            
            # Run performance tests if requested
            performance_results = None
            if performance:
                performance_results = self.run_performance_tests(verbose)
            
            # Calculate totals
            total_tests_run = sum(result['tests_run'] for result in module_results.values())
            total_tests_passed = sum(result['tests_passed'] for result in module_results.values())
            total_tests_failed = sum(result['tests_failed'] for result in module_results.values())
            
            if integration_results:
                total_tests_run += integration_results['tests_run']
                total_tests_passed += integration_results['tests_passed']
                total_tests_failed += integration_results['tests_failed']
            
            if performance_results:
                total_tests_run += performance_results['tests_run']
                total_tests_passed += performance_results['tests_passed']
                total_tests_failed += performance_results['tests_failed']
            
            # Check overall success
            all_module_success = all(result['success'] for result in module_results.values())
            integration_success = integration_results['success'] if integration_results else True
            performance_success = performance_results['success'] if performance_results else True
            
            overall_success = all_module_success and integration_success and performance_success
            
            self.end_time = time.time()
            total_duration = self.end_time - self.start_time
            
            return {
                'success': overall_success,
                'total_tests_run': total_tests_run,
                'total_tests_passed': total_tests_passed,
                'total_tests_failed': total_tests_failed,
                'total_duration': total_duration,
                'module_results': module_results,
                'integration_results': integration_results,
                'performance_results': performance_results
            }
            
        except Exception as e:
            logger.error(f"Test run failed: {e}")
            self.end_time = time.time()
            return {
                'success': False,
                'error': str(e),
                'total_tests_run': 0,
                'total_tests_passed': 0,
                'total_tests_failed': 0,
                'total_duration': self.end_time - self.start_time if self.start_time else 0
            }
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results."""
        print("\n" + "="*60)
        print("LERK SYSTEM TEST RESULTS")
        print("="*60)
        
        # Overall summary
        print(f"Overall Success: {'✓' if results['success'] else '✗'}")
        print(f"Total Tests: {results['total_tests_run']}")
        print(f"Passed: {results['total_tests_passed']}")
        print(f"Failed: {results['total_tests_failed']}")
        print(f"Duration: {results['total_duration']:.2f} seconds")
        
        # Module results
        print(f"\nModule Results:")
        for module, result in results['module_results'].items():
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {module}: {result['tests_passed']}/{result['tests_run']} passed")
            if not result['success'] and 'error' in result:
                print(f"    Error: {result['error']}")
        
        # Integration results
        if results['integration_results']:
            integration = results['integration_results']
            status = "✓" if integration['success'] else "✗"
            print(f"  {status} Integration Tests: {integration['tests_passed']}/{integration['tests_run']} passed")
            if not integration['success'] and 'error' in integration:
                print(f"    Error: {integration['error']}")
        
        # Performance results
        if results['performance_results']:
            performance = results['performance_results']
            status = "✓" if performance['success'] else "✗"
            print(f"  {status} Performance Tests: {performance['tests_passed']}/{performance['tests_run']} passed")
            if not performance['success'] and 'error' in performance:
                print(f"    Error: {performance['error']}")
        
        print("="*60)
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save test results to file."""
        try:
            # Add timestamp
            results['timestamp'] = time.time()
            results['start_time'] = self.start_time
            results['end_time'] = self.end_time
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='LERK Test Runner')
    parser.add_argument('--modules', nargs='+', 
                       help='Specific modules to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--coverage', action='store_true',
                       help='Enable coverage reporting')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--output', '-o',
                       help='Output file for results')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test runner
    runner = TestRunner(project_root)
    
    try:
        # Run tests
        results = runner.run_all_tests(
            modules=args.modules,
            verbose=args.verbose,
            coverage=args.coverage,
            integration=args.integration,
            performance=args.performance
        )
        
        # Output results
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            runner.print_results(results)
        
        # Save results if requested
        if args.output:
            runner.save_results(results, args.output)
        
        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
