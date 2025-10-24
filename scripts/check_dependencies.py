#!/usr/bin/env python3
"""
LERK System - Dependency Check Script
This script checks if all required dependencies are installed and configured.
"""

import argparse
import logging
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dependency_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DependencyChecker:
    """Checks system dependencies and configuration."""
    
    def __init__(self):
        """Initialize the dependency checker."""
        self.project_root = Path(__file__).parent.parent
        self.required_packages = [
            'langchain',
            'langchain_openai',
            'pydantic',
            'unstructured',
            'sentence_transformers',
            'qdrant_client',
            'sqlalchemy',
            'pandas',
            'numpy',
            'torch',
            'transformers',
            'bertopic',
            'umap',
            'hdbscan',
            'pytest',
            'pytest_cov'
        ]
        
        self.optional_packages = [
            'jupyter',
            'matplotlib',
            'seaborn',
            'scikit_learn'
        ]
        
        self.system_commands = [
            'python3',
            'pip',
            'git'
        ]
        
        self.optional_commands = [
            'docker',
            'docker-compose',
            'postgres',
            'redis-server',
            'qdrant'
        ]
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version."""
        try:
            import sys
            version = sys.version_info
            if version.major == 3 and version.minor >= 8:
                return True, f"Python {version.major}.{version.minor}.{version.micro}"
            else:
                return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"
        except Exception as e:
            return False, f"Failed to check Python version: {e}"
    
    def check_package(self, package_name: str) -> Tuple[bool, str]:
        """Check if a Python package is installed."""
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            return True, f"{package_name} {version}"
        except ImportError:
            return False, f"{package_name} not installed"
        except Exception as e:
            return False, f"{package_name} error: {e}"
    
    def check_command(self, command: str) -> Tuple[bool, str]:
        """Check if a system command is available."""
        try:
            result = subprocess.run(
                [command, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                return True, f"{command} {version}"
            else:
                return False, f"{command} not found"
        except subprocess.TimeoutExpired:
            return False, f"{command} timeout"
        except FileNotFoundError:
            return False, f"{command} not found"
        except Exception as e:
            return False, f"{command} error: {e}"
    
    def check_environment_variables(self) -> Dict[str, Tuple[bool, str]]:
        """Check environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'OpenAI API key for LLM operations',
            'DATABASE_URL': 'Database connection URL',
            'QDRANT_URL': 'Qdrant vector database URL',
            'REDIS_URL': 'Redis cache URL'
        }
        
        results = {}
        for var, description in env_vars.items():
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if 'key' in var.lower() or 'password' in var.lower():
                    masked_value = value[:8] + '...' if len(value) > 8 else '***'
                    results[var] = (True, f"{var}={masked_value}")
                else:
                    results[var] = (True, f"{var}={value}")
            else:
                results[var] = (False, f"{var} not set ({description})")
        
        return results
    
    def check_database_connection(self) -> Tuple[bool, str]:
        """Check database connection."""
        try:
            database_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/lerk_db')
            
            from sqlalchemy import create_engine, text
            engine = create_engine(database_url, echo=False)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True, "Database connection successful"
                
        except Exception as e:
            return False, f"Database connection failed: {e}"
    
    def check_vector_db_connection(self) -> Tuple[bool, str]:
        """Check vector database connection."""
        try:
            qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
            
            import requests
            response = requests.get(f"{qdrant_url}/health", timeout=5)
            if response.status_code == 200:
                return True, "Vector database connection successful"
            else:
                return False, f"Vector database returned status {response.status_code}"
                
        except Exception as e:
            return False, f"Vector database connection failed: {e}"
    
    def check_redis_connection(self) -> Tuple[bool, str]:
        """Check Redis connection."""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            return True, "Redis connection successful"
            
        except Exception as e:
            return False, f"Redis connection failed: {e}"
    
    def check_file_permissions(self) -> Dict[str, Tuple[bool, str]]:
        """Check file permissions."""
        paths_to_check = [
            'logs',
            'data',
            'temp',
            'config'
        ]
        
        results = {}
        for path in paths_to_check:
            full_path = self.project_root / path
            try:
                if full_path.exists():
                    if os.access(full_path, os.R_OK | os.W_OK):
                        results[path] = (True, f"{path} accessible")
                    else:
                        results[path] = (False, f"{path} permission denied")
                else:
                    # Try to create directory
                    full_path.mkdir(parents=True, exist_ok=True)
                    results[path] = (True, f"{path} created")
            except Exception as e:
                results[path] = (False, f"{path} error: {e}")
        
        return results
    
    def check_docker_services(self) -> Dict[str, Tuple[bool, str]]:
        """Check Docker services."""
        services = ['postgres', 'redis', 'qdrant']
        results = {}
        
        for service in services:
            try:
                result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={service}', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and service in result.stdout:
                    results[service] = (True, f"{service} container running")
                else:
                    results[service] = (False, f"{service} container not running")
                    
            except Exception as e:
                results[service] = (False, f"{service} check failed: {e}")
        
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive dependency check."""
        results = {
            'python_version': self.check_python_version(),
            'required_packages': {},
            'optional_packages': {},
            'system_commands': {},
            'optional_commands': {},
            'environment_variables': {},
            'database_connection': (False, 'Not checked'),
            'vector_db_connection': (False, 'Not checked'),
            'redis_connection': (False, 'Not checked'),
            'file_permissions': {},
            'docker_services': {}
        }
        
        # Check required packages
        for package in self.required_packages:
            results['required_packages'][package] = self.check_package(package)
        
        # Check optional packages
        for package in self.optional_packages:
            results['optional_packages'][package] = self.check_package(package)
        
        # Check system commands
        for command in self.system_commands:
            results['system_commands'][command] = self.check_command(command)
        
        # Check optional commands
        for command in self.optional_commands:
            results['optional_commands'][command] = self.check_command(command)
        
        # Check environment variables
        results['environment_variables'] = self.check_environment_variables()
        
        # Check file permissions
        results['file_permissions'] = self.check_file_permissions()
        
        # Check Docker services
        results['docker_services'] = self.check_docker_services()
        
        # Check connections (if environment variables are set)
        if os.getenv('DATABASE_URL'):
            results['database_connection'] = self.check_database_connection()
        
        if os.getenv('QDRANT_URL'):
            results['vector_db_connection'] = self.check_vector_db_connection()
        
        if os.getenv('REDIS_URL'):
            results['redis_connection'] = self.check_redis_connection()
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print dependency check results."""
        print("\n" + "="*60)
        print("LERK SYSTEM DEPENDENCY CHECK")
        print("="*60)
        
        # Python version
        success, message = results['python_version']
        status = "✓" if success else "✗"
        print(f"{status} Python: {message}")
        
        # Required packages
        print(f"\nRequired Packages:")
        for package, (success, message) in results['required_packages'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {message}")
        
        # Optional packages
        print(f"\nOptional Packages:")
        for package, (success, message) in results['optional_packages'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {message}")
        
        # System commands
        print(f"\nSystem Commands:")
        for command, (success, message) in results['system_commands'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {message}")
        
        # Optional commands
        print(f"\nOptional Commands:")
        for command, (success, message) in results['optional_commands'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {message}")
        
        # Environment variables
        print(f"\nEnvironment Variables:")
        for var, (success, message) in results['environment_variables'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {message}")
        
        # File permissions
        print(f"\nFile Permissions:")
        for path, (success, message) in results['file_permissions'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {message}")
        
        # Docker services
        print(f"\nDocker Services:")
        for service, (success, message) in results['docker_services'].items():
            status = "✓" if success else "✗"
            print(f"  {status} {message}")
        
        # Connections
        print(f"\nService Connections:")
        for connection_type in ['database_connection', 'vector_db_connection', 'redis_connection']:
            success, message = results[connection_type]
            status = "✓" if success else "✗"
            print(f"  {status} {connection_type.replace('_', ' ').title()}: {message}")
        
        print("="*60)
    
    def get_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of dependency check."""
        total_required = len(self.required_packages)
        total_optional = len(self.optional_packages)
        total_system = len(self.system_commands)
        total_optional_cmd = len(self.optional_commands)
        
        required_installed = sum(1 for success, _ in results['required_packages'].values() if success)
        optional_installed = sum(1 for success, _ in results['optional_packages'].values() if success)
        system_available = sum(1 for success, _ in results['system_commands'].values() if success)
        optional_available = sum(1 for success, _ in results['optional_commands'].values() if success)
        
        env_set = sum(1 for success, _ in results['environment_variables'].values() if success)
        total_env = len(results['environment_variables'])
        
        file_ok = sum(1 for success, _ in results['file_permissions'].values() if success)
        total_files = len(results['file_permissions'])
        
        docker_ok = sum(1 for success, _ in results['docker_services'].values() if success)
        total_docker = len(results['docker_services'])
        
        connections_ok = sum(1 for success, _ in [
            results['database_connection'],
            results['vector_db_connection'],
            results['redis_connection']
        ] if success)
        
        return {
            'required_packages': f"{required_installed}/{total_required}",
            'optional_packages': f"{optional_installed}/{total_optional}",
            'system_commands': f"{system_available}/{total_system}",
            'optional_commands': f"{optional_available}/{total_optional_cmd}",
            'environment_variables': f"{env_set}/{total_env}",
            'file_permissions': f"{file_ok}/{total_files}",
            'docker_services': f"{docker_ok}/{total_docker}",
            'connections': f"{connections_ok}/3"
        }


def main():
    """Main entry point for dependency check."""
    parser = argparse.ArgumentParser(description='LERK Dependency Check')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create dependency checker
    checker = DependencyChecker()
    
    try:
        # Run comprehensive check
        results = checker.run_comprehensive_check()
        
        if args.json:
            # Output JSON results
            import json
            print(json.dumps(results, indent=2))
        else:
            # Print formatted results
            checker.print_results(results)
            
            # Print summary
            summary = checker.get_summary(results)
            print(f"\nSUMMARY:")
            print(f"  Required packages: {summary['required_packages']}")
            print(f"  Optional packages: {summary['optional_packages']}")
            print(f"  System commands: {summary['system_commands']}")
            print(f"  Optional commands: {summary['optional_commands']}")
            print(f"  Environment variables: {summary['environment_variables']}")
            print(f"  File permissions: {summary['file_permissions']}")
            print(f"  Docker services: {summary['docker_services']}")
            print(f"  Service connections: {summary['connections']}")
            
            # Check if all required dependencies are met
            required_ok = all(success for success, _ in results['required_packages'].values())
            system_ok = all(success for success, _ in results['system_commands'].values())
            
            if required_ok and system_ok:
                print("\n✓ All required dependencies are satisfied!")
                sys.exit(0)
            else:
                print("\n✗ Some required dependencies are missing!")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
