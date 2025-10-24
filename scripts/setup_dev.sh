#!/bin/bash

# LERK System - Development Environment Setup Script
# This script sets up the development environment for the LERK System

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create project structure
create_project_structure() {
    print_status "Creating project structure..."
    
    # Create necessary directories
    mkdir -p data/{raw,processed,output}
    mkdir -p logs
    mkdir -p temp
    mkdir -p config
    mkdir -p tests/{ingest_test,enrichment_test,logic_extractor_test,clustering_test,consolidation_test,retriever_test,qa_agent_test}
    
    print_success "Project structure created"
}

# Function to create development configuration
create_dev_config() {
    print_status "Creating development configuration..."
    
    # Create config/dev.yaml
    cat > config/dev.yaml << EOF
# LERK System - Development Configuration

# Database Configuration
database:
  url: "postgresql://user:password@localhost:5432/lerk_dev"
  echo: true
  pool_size: 5
  max_overflow: 10

# Vector Database Configuration
vector_db:
  url: "http://localhost:6333"
  collection_name: "lerk_dev"
  vector_size: 384

# Redis Configuration
redis:
  url: "redis://localhost:6379/0"
  max_connections: 10

# Logging Configuration
logging:
  level: "DEBUG"
  format: "json"
  file: "logs/lerk_dev.log"
  max_size: "10MB"
  backup_count: 5

# Processing Configuration
processing:
  max_workers: 2
  chunk_size: 500
  chunk_overlap: 100
  batch_size: 10

# API Configuration
api:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  reload: true
  log_level: "debug"

# Development Settings
development:
  debug: true
  testing: true
  mock_llm: false
  cache_responses: true
EOF

    print_success "Development configuration created"
}

# Function to create test configuration
create_test_config() {
    print_status "Creating test configuration..."
    
    # Create config/test.yaml
    cat > config/test.yaml << EOF
# LERK System - Test Configuration

# Database Configuration
database:
  url: "postgresql://user:password@localhost:5432/lerk_test"
  echo: false
  pool_size: 1
  max_overflow: 0

# Vector Database Configuration
vector_db:
  url: "http://localhost:6333"
  collection_name: "lerk_test"
  vector_size: 384

# Redis Configuration
redis:
  url: "redis://localhost:6379/1"
  max_connections: 1

# Logging Configuration
logging:
  level: "WARNING"
  format: "simple"
  file: "logs/lerk_test.log"

# Processing Configuration
processing:
  max_workers: 1
  chunk_size: 100
  chunk_overlap: 20
  batch_size: 1

# API Configuration
api:
  host: "127.0.0.1"
  port: 8001
  workers: 1
  reload: false
  log_level: "warning"

# Test Settings
testing:
  debug: false
  testing: true
  mock_llm: true
  cache_responses: false
  cleanup_after_tests: true
EOF

    print_success "Test configuration created"
}

# Function to create Docker Compose for development
create_docker_compose() {
    print_status "Creating Docker Compose for development..."
    
    cat > docker-compose.dev.yml << EOF
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: lerk_dev
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d lerk_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # LERK API Service (Development)
  lerk-api:
    build:
      context: .
      dockerfile: services/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/lerk_dev
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
      - LOG_LEVEL=DEBUG
    volumes:
      - .:/app
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    command: ["python", "-m", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
EOF

    print_success "Docker Compose configuration created"
}

# Function to create pre-commit hooks
create_pre_commit_hooks() {
    print_status "Setting up pre-commit hooks..."
    
    # Create .pre-commit-config.yaml
    cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length", "88", "--extend-ignore", "E203"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

    print_success "Pre-commit hooks configuration created"
}

# Function to create development scripts
create_dev_scripts() {
    print_status "Creating development scripts..."
    
    # Create scripts/dev_start.sh
    cat > scripts/dev_start.sh << 'EOF'
#!/bin/bash
# Start development environment

echo "Starting LERK development environment..."

# Start Docker services
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo "Checking service health..."
docker-compose -f docker-compose.dev.yml ps

echo "Development environment started!"
echo "API available at: http://localhost:8000"
echo "Database available at: localhost:5432"
echo "Redis available at: localhost:6379"
echo "Qdrant available at: http://localhost:6333"
EOF

    # Create scripts/dev_stop.sh
    cat > scripts/dev_stop.sh << 'EOF'
#!/bin/bash
# Stop development environment

echo "Stopping LERK development environment..."

# Stop Docker services
docker-compose -f docker-compose.dev.yml down

echo "Development environment stopped!"
EOF

    # Create scripts/dev_clean.sh
    cat > scripts/dev_clean.sh << 'EOF'
#!/bin/bash
# Clean development environment

echo "Cleaning LERK development environment..."

# Stop and remove containers
docker-compose -f docker-compose.dev.yml down -v

# Remove volumes
docker volume rm lerk_system_postgres_data lerk_system_redis_data lerk_system_qdrant_data 2>/dev/null || true

# Clean up logs
rm -rf logs/*

# Clean up temp files
rm -rf temp/*

echo "Development environment cleaned!"
EOF

    # Make scripts executable
    chmod +x scripts/dev_start.sh
    chmod +x scripts/dev_stop.sh
    chmod +x scripts/dev_clean.sh

    print_success "Development scripts created"
}

# Function to create VS Code configuration
create_vscode_config() {
    print_status "Creating VS Code configuration..."
    
    mkdir -p .vscode
    
    # Create .vscode/settings.json
    cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=88"],
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "python.testing.unittestEnabled": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true,
        "**/htmlcov": true,
        "**/.pytest_cache": true
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
EOF

    # Create .vscode/launch.json
    cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "LERK API",
            "type": "python",
            "request": "launch",
            "program": "\${workspaceFolder}/services/api/main.py",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}",
                "DATABASE_URL": "postgresql://user:password@localhost:5432/lerk_dev",
                "REDIS_URL": "redis://localhost:6379/0",
                "QDRANT_URL": "http://localhost:6333"
            }
        },
        {
            "name": "LERK Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}"
            }
        }
    ]
}
EOF

    print_success "VS Code configuration created"
}

# Function to create Jupyter configuration
create_jupyter_config() {
    print_status "Creating Jupyter configuration..."
    
    # Create jupyter_notebook_config.py
    cat > jupyter_notebook_config.py << EOF
# Jupyter Notebook Configuration for LERK System

c = get_config()

# Set the IP address to listen on
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

# Set the notebook directory
c.NotebookApp.notebook_dir = '/Users/haydenchiu/git/LERK_System'

# Enable extensions
c.NotebookApp.nbserver_extensions = {
    'jupyterlab': True,
}

# Security settings
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.disable_check_xsrf = True

# Allow all origins (development only)
c.NotebookApp.allow_origin = '*'
EOF

    print_success "Jupyter configuration created"
}

# Main setup function
main() {
    echo "=========================================="
    echo "LERK System - Development Environment Setup"
    echo "=========================================="
    
    # Create project structure
    create_project_structure
    
    # Create configurations
    create_dev_config
    create_test_config
    
    # Create Docker Compose
    create_docker_compose
    
    # Create pre-commit hooks
    create_pre_commit_hooks
    
    # Create development scripts
    create_dev_scripts
    
    # Create VS Code configuration
    create_vscode_config
    
    # Create Jupyter configuration
    create_jupyter_config
    
    echo "=========================================="
    print_success "Development environment setup completed!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run: ./scripts/install_deps.sh"
    echo "2. Run: ./scripts/dev_start.sh"
    echo "3. Run: python scripts/init_db.py"
    echo "4. Start developing!"
    echo ""
}

# Run main function
main "$@"
