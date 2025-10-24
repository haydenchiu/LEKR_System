#!/usr/bin/env python3
"""
LERK System - Coverage Report Script
This script generates comprehensive coverage reports for the LERK System.
"""

import argparse
import logging
import sys
import subprocess
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
        logging.FileHandler('logs/coverage_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CoverageReporter:
    """Generates comprehensive coverage reports for the LERK System."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the coverage reporter.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.modules = [
            'ingest',
            'enrichment',
            'logic_extractor',
            'clustering',
            'consolidation',
            'retriever',
            'qa_agent'
        ]
        
        self.coverage_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_coverage_for_module(self, module: str, verbose: bool = False) -> Dict[str, Any]:
        """Run coverage analysis for a specific module."""
        try:
            logger.info(f"Running coverage analysis for {module}")
            
            test_dir = self.project_root / 'tests' / f'{module}_test'
            if not test_dir.exists():
                return {
                    'success': False,
                    'error': f"Test directory not found: {test_dir}",
                    'coverage_percent': 0,
                    'lines_covered': 0,
                    'lines_total': 0,
                    'duration': 0
                }
            
            # Build coverage command
            cmd = [
                'python', '-m', 'pytest',
                str(test_dir),
                '--cov', module,
                '--cov-report=json',
                '--cov-report=term-missing',
                '--cov-report=html',
                '--cov-branch',
                '--cov-fail-under=80'
            ]
            
            if verbose:
                cmd.append('-v')
            
            # Run coverage analysis
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Parse coverage results
            coverage_data = self._parse_coverage_output(result.stdout, result.stderr)
            
            return {
                'success': result.returncode == 0,
                'coverage_percent': coverage_data['coverage_percent'],
                'lines_covered': coverage_data['lines_covered'],
                'lines_total': coverage_data['lines_total'],
                'branches_covered': coverage_data['branches_covered'],
                'branches_total': coverage_data['branches_total'],
                'duration': duration,
                'output': result.stdout,
                'error_output': result.stderr,
                'coverage_data': coverage_data
            }
            
        except Exception as e:
            logger.error(f"Failed to run coverage for {module}: {e}")
            return {
                'success': False,
                'error': str(e),
                'coverage_percent': 0,
                'lines_covered': 0,
                'lines_total': 0,
                'duration': 0
            }
    
    def _parse_coverage_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse coverage output to extract metrics."""
        try:
            # Look for coverage percentage in output
            coverage_percent = 0
            lines_covered = 0
            lines_total = 0
            branches_covered = 0
            branches_total = 0
            
            # Parse from stdout
            for line in stdout.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    # Extract coverage percentage
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            try:
                                coverage_percent = float(part.replace('%', ''))
                                break
                            except ValueError:
                                continue
                
                # Extract line counts
                if 'lines' in line.lower() and 'covered' in line.lower():
                    try:
                        # Format: "lines: 85% (123/145)"
                        if '(' in line and ')' in line:
                            numbers = line.split('(')[1].split(')')[0]
                            if '/' in numbers:
                                lines_covered, lines_total = map(int, numbers.split('/'))
                    except ValueError:
                        continue
                
                # Extract branch counts
                if 'branches' in line.lower() and 'covered' in line.lower():
                    try:
                        # Format: "branches: 75% (45/60)"
                        if '(' in line and ')' in line:
                            numbers = line.split('(')[1].split(')')[0]
                            if '/' in numbers:
                                branches_covered, branches_total = map(int, numbers.split('/'))
                    except ValueError:
                        continue
            
            return {
                'coverage_percent': coverage_percent,
                'lines_covered': lines_covered,
                'lines_total': lines_total,
                'branches_covered': branches_covered,
                'branches_total': branches_total
            }
            
        except Exception as e:
            logger.error(f"Failed to parse coverage output: {e}")
            return {
                'coverage_percent': 0,
                'lines_covered': 0,
                'lines_total': 0,
                'branches_covered': 0,
                'branches_total': 0
            }
    
    def run_combined_coverage(self, modules: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Run combined coverage analysis for multiple modules."""
        try:
            logger.info("Running combined coverage analysis")
            
            # Build combined coverage command
            cmd = [
                'python', '-m', 'pytest',
                '--cov', '--cov-report=json',
                '--cov-report=term-missing',
                '--cov-report=html',
                '--cov-branch',
                '--cov-fail-under=80'
            ]
            
            # Add test directories
            for module in modules:
                test_dir = self.project_root / 'tests' / f'{module}_test'
                if test_dir.exists():
                    cmd.append(str(test_dir))
            
            if verbose:
                cmd.append('-v')
            
            # Run combined coverage
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Parse combined results
            coverage_data = self._parse_coverage_output(result.stdout, result.stderr)
            
            return {
                'success': result.returncode == 0,
                'coverage_percent': coverage_data['coverage_percent'],
                'lines_covered': coverage_data['lines_covered'],
                'lines_total': coverage_data['lines_total'],
                'branches_covered': coverage_data['branches_covered'],
                'branches_total': coverage_data['branches_total'],
                'duration': duration,
                'output': result.stdout,
                'error_output': result.stderr,
                'coverage_data': coverage_data
            }
            
        except Exception as e:
            logger.error(f"Combined coverage analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'coverage_percent': 0,
                'lines_covered': 0,
                'lines_total': 0,
                'duration': 0
            }
    
    def generate_coverage_report(self, 
                               modules: Optional[List[str]] = None,
                               verbose: bool = False,
                               combined: bool = False) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        self.start_time = time.time()
        
        try:
            logger.info("Starting coverage analysis")
            
            # Determine which modules to analyze
            test_modules = modules or self.modules
            
            if combined:
                # Run combined coverage analysis
                combined_results = self.run_combined_coverage(test_modules, verbose)
                
                return {
                    'success': combined_results['success'],
                    'combined_coverage': combined_results,
                    'module_results': {},
                    'total_duration': combined_results['duration']
                }
            else:
                # Run individual module coverage
                module_results = {}
                for module in test_modules:
                    if module in self.modules:
                        module_results[module] = self.run_coverage_for_module(module, verbose)
                    else:
                        module_results[module] = {
                            'success': False,
                            'error': f"Unknown module: {module}",
                            'coverage_percent': 0,
                            'lines_covered': 0,
                            'lines_total': 0,
                            'duration': 0
                        }
                
                # Calculate totals
                total_lines_covered = sum(result['lines_covered'] for result in module_results.values())
                total_lines_total = sum(result['lines_total'] for result in module_results.values())
                total_branches_covered = sum(result.get('branches_covered', 0) for result in module_results.values())
                total_branches_total = sum(result.get('branches_total', 0) for result in module_results.values())
                
                overall_coverage = (total_lines_covered / total_lines_total * 100) if total_lines_total > 0 else 0
                overall_branch_coverage = (total_branches_covered / total_branches_total * 100) if total_branches_total > 0 else 0
                
                # Check if all modules meet coverage threshold
                all_modules_success = all(result['success'] for result in module_results.values())
                coverage_threshold_met = overall_coverage >= 80
                
                self.end_time = time.time()
                total_duration = self.end_time - self.start_time
                
                return {
                    'success': all_modules_success and coverage_threshold_met,
                    'overall_coverage': overall_coverage,
                    'overall_branch_coverage': overall_branch_coverage,
                    'total_lines_covered': total_lines_covered,
                    'total_lines_total': total_lines_total,
                    'total_branches_covered': total_branches_covered,
                    'total_branches_total': total_branches_total,
                    'module_results': module_results,
                    'total_duration': total_duration
                }
                
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            self.end_time = time.time()
            return {
                'success': False,
                'error': str(e),
                'total_duration': self.end_time - self.start_time if self.start_time else 0
            }
    
    def print_results(self, results: Dict[str, Any]):
        """Print coverage results."""
        print("\n" + "="*60)
        print("LERK SYSTEM COVERAGE REPORT")
        print("="*60)
        
        if 'combined_coverage' in results:
            # Combined coverage results
            combined = results['combined_coverage']
            print(f"Overall Success: {'✓' if combined['success'] else '✗'}")
            print(f"Coverage: {combined['coverage_percent']:.1f}%")
            print(f"Lines: {combined['lines_covered']}/{combined['lines_total']}")
            print(f"Branches: {combined['branches_covered']}/{combined['branches_total']}")
            print(f"Duration: {combined['duration']:.2f} seconds")
            
        else:
            # Individual module results
            print(f"Overall Success: {'✓' if results['success'] else '✗'}")
            print(f"Overall Coverage: {results['overall_coverage']:.1f}%")
            print(f"Overall Branch Coverage: {results['overall_branch_coverage']:.1f}%")
            print(f"Total Lines: {results['total_lines_covered']}/{results['total_lines_total']}")
            print(f"Total Branches: {results['total_branches_covered']}/{results['total_branches_total']}")
            print(f"Duration: {results['total_duration']:.2f} seconds")
            
            # Module breakdown
            print(f"\nModule Coverage:")
            for module, result in results['module_results'].items():
                status = "✓" if result['success'] else "✗"
                coverage = result['coverage_percent']
                lines = f"{result['lines_covered']}/{result['lines_total']}"
                print(f"  {status} {module}: {coverage:.1f}% ({lines} lines)")
                if not result['success'] and 'error' in result:
                    print(f"    Error: {result['error']}")
        
        print("="*60)
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save coverage results to file."""
        try:
            # Add timestamp
            results['timestamp'] = time.time()
            results['start_time'] = self.start_time
            results['end_time'] = self.end_time
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Coverage results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def generate_html_report(self, output_dir: str = "htmlcov"):
        """Generate HTML coverage report."""
        try:
            logger.info(f"Generating HTML coverage report in {output_dir}")
            
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Run coverage HTML report
            cmd = [
                'python', '-m', 'coverage', 'html',
                '-d', output_dir,
                '--skip-covered'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            if result.returncode == 0:
                logger.info(f"HTML coverage report generated in {output_dir}")
                return True
            else:
                logger.error(f"Failed to generate HTML report: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            return False


def main():
    """Main entry point for coverage reporter."""
    parser = argparse.ArgumentParser(description='LERK Coverage Reporter')
    parser.add_argument('--modules', nargs='+', 
                       help='Specific modules to analyze')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--combined', action='store_true',
                       help='Run combined coverage analysis')
    parser.add_argument('--output', '-o',
                       help='Output file for results')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--html-dir', default='htmlcov',
                       help='Directory for HTML report')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create coverage reporter
    reporter = CoverageReporter(project_root)
    
    try:
        # Generate coverage report
        results = reporter.generate_coverage_report(
            modules=args.modules,
            verbose=args.verbose,
            combined=args.combined
        )
        
        # Output results
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            reporter.print_results(results)
        
        # Save results if requested
        if args.output:
            reporter.save_results(results, args.output)
        
        # Generate HTML report if requested
        if args.html:
            reporter.generate_html_report(args.html_dir)
        
        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Coverage reporter failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
