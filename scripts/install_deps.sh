#!/bin/bash

# LERK System - Dependency Installation Script
# This script installs all required dependencies for the LERK System

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

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_status "Found Python $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible (>= 3.8)"
        else
            print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Function to create virtual environment
create_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Please run setup_dev.sh first."
        exit 1
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "pip upgraded"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Function to install system dependencies (for unstructured)
install_system_deps() {
    print_status "Installing system dependencies for document processing..."
    
    # Check OS and install appropriate dependencies
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        if command_exists apt-get; then
            print_status "Installing system dependencies for Linux..."
            sudo apt-get update
            sudo apt-get install -y \
                tesseract-ocr \
                libtesseract-dev \
                poppler-utils \
                libmagic1 \
                libmagic-dev \
                libxml2-dev \
                libxslt1-dev \
                libffi-dev \
                libssl-dev \
                libjpeg-dev \
                libpng-dev \
                libtiff-dev \
                libopenjp2-7-dev \
                zlib1g-dev \
                libfreetype6-dev \
                liblcms2-dev \
                libwebp-dev \
                libharfbuzz-dev \
                libfribidi-dev \
                libxcb1-dev
        else
            print_warning "System dependencies may need to be installed manually for document processing"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            print_status "Installing system dependencies for macOS..."
            brew install tesseract poppler libmagic
        else
            print_warning "Homebrew not found. System dependencies may need to be installed manually"
        fi
    else
        print_warning "Unsupported OS. System dependencies may need to be installed manually"
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test Python imports
    python3 -c "
import sys
try:
    import langchain
    import langchain_openai
    import pydantic
    import unstructured
    import sentence_transformers
    import qdrant_client
    import sqlalchemy
    import pandas
    import numpy
    import torch
    import transformers
    import bertopic
    import umap
    import hdbscan
    import pytest
    print('✓ All core dependencies imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "All dependencies verified successfully"
    else
        print_error "Dependency verification failed"
        exit 1
    fi
}

# Function to create .env file if it doesn't exist
create_env_file() {
    if [ ! -f ".env" ]; then
        print_status "Creating .env file..."
        cat > .env << EOF
# LERK System Environment Configuration

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/lerk_db
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=lerk_db
DATABASE_USER=user
DATABASE_PASSWORD=password

# Vector Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Redis Configuration (for caching)
REDIS_URL=redis://localhost:6379/0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Processing Configuration
MAX_WORKERS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Clustering Configuration
CLUSTERING_MIN_CLUSTER_SIZE=5
CLUSTERING_MIN_SAMPLES=3

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Development Configuration
DEBUG=False
TESTING=False
EOF
        print_success ".env file created"
        print_warning "Please edit .env file with your actual configuration values"
    else
        print_warning ".env file already exists"
    fi
}

# Main installation function
main() {
    echo "=========================================="
    echo "LERK System - Dependency Installation"
    echo "=========================================="
    
    # Check Python version
    check_python_version
    
    # Create virtual environment
    create_venv
    
    # Activate virtual environment
    activate_venv
    
    # Upgrade pip
    upgrade_pip
    
    # Install system dependencies
    install_system_deps
    
    # Install Python dependencies
    install_python_deps
    
    # Create .env file
    create_env_file
    
    # Verify installation
    verify_installation
    
    echo "=========================================="
    print_success "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Run: source venv/bin/activate"
    echo "3. Run: python scripts/setup_dev.py"
    echo "4. Run: python scripts/init_db.py"
    echo ""
}

# Run main function
main "$@"
