# LERK System Scripts

This directory contains helper scripts for the LERK (Logic Extraction and Reasoning Knowledge) System. These scripts provide essential functionality for development, testing, deployment, and maintenance.

## 📁 Script Categories

### 🚀 **Development Setup Scripts**
- **`install_deps.sh`** - Install all required dependencies
- **`setup_dev.sh`** - Set up development environment
- **`setup_env.py`** - Configure environment variables and settings
- **`check_dependencies.py`** - Verify system dependencies

### 🔄 **Pipeline Orchestration Scripts**
- **`run_ingestion_pipeline.py`** - Run complete document ingestion pipeline
- **`run_consolidation_pipeline.py`** - Run knowledge consolidation pipeline

### 🗄️ **Database Management Scripts**
- **`init_db.py`** - Initialize database schema and tables
- **`migrate_db.py`** - Handle database migrations and schema updates

### 🧪 **Testing and Quality Scripts**
- **`run_all_tests.py`** - Run comprehensive test suite
- **`coverage_report.py`** - Generate code coverage reports

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Install dependencies
./scripts/install_deps.sh

# Set up development environment
./scripts/setup_dev.sh

# Configure environment
python scripts/setup_env.py

# Check dependencies
python scripts/check_dependencies.py
```

### 2. Database Setup
```bash
# Initialize database
python scripts/init_db.py

# Run migrations (if needed)
python scripts/migrate_db.py --migrate-to latest
```

### 3. Run Tests
```bash
# Run all tests
python scripts/run_all_tests.py

# Generate coverage report
python scripts/coverage_report.py --html
```

### 4. Run Pipelines
```bash
# Document ingestion pipeline
python scripts/run_ingestion_pipeline.py data/input data/output

# Knowledge consolidation pipeline
python scripts/run_consolidation_pipeline.py data/output data/consolidated
```

## 📋 Detailed Script Documentation

### Development Setup Scripts

#### `install_deps.sh`
Installs all required dependencies for the LERK System.

**Features:**
- Python version checking (requires 3.8+)
- Virtual environment creation
- System dependency installation
- Python package installation
- Environment file creation
- Installation verification

**Usage:**
```bash
./scripts/install_deps.sh
```

**Options:**
- Automatically detects OS and installs appropriate system dependencies
- Creates `.env` file with default configuration
- Verifies all installations

#### `setup_dev.sh`
Sets up the complete development environment.

**Features:**
- Project structure creation
- Configuration file generation
- Docker Compose setup
- Pre-commit hooks configuration
- VS Code configuration
- Jupyter configuration

**Usage:**
```bash
./scripts/setup_dev.sh
```

**Creates:**
- `config/dev.yaml` - Development configuration
- `config/test.yaml` - Test configuration
- `docker-compose.dev.yml` - Development Docker setup
- `.pre-commit-config.yaml` - Pre-commit hooks
- `.vscode/` - VS Code configuration

#### `setup_env.py`
Configures environment variables and settings.

**Features:**
- Environment file creation
- Configuration file generation
- Docker Compose setup
- Pre-commit hooks
- VS Code configuration

**Usage:**
```bash
python scripts/setup_env.py [--config config_file] [--verbose]
```

**Options:**
- `--config` - Path to configuration file
- `--verbose` - Enable verbose logging

#### `check_dependencies.py`
Verifies system dependencies and configuration.

**Features:**
- Python package verification
- System command checking
- Environment variable validation
- Database connection testing
- File permission checking
- Docker service verification

**Usage:**
```bash
python scripts/check_dependencies.py [--verbose] [--json]
```

**Options:**
- `--verbose` - Enable verbose logging
- `--json` - Output results in JSON format

### Pipeline Orchestration Scripts

#### `run_ingestion_pipeline.py`
Runs the complete document ingestion pipeline.

**Features:**
- Document parsing and chunking
- Chunk enrichment
- Logic extraction
- Document consolidation
- Parallel processing support
- Async processing support
- Comprehensive statistics

**Usage:**
```bash
python scripts/run_ingestion_pipeline.py input_path output_path [options]
```

**Options:**
- `--config` - Configuration preset (default, fast, high_quality, large_document)
- `--workers` - Maximum number of worker threads
- `--no-async` - Disable async processing
- `--verbose` - Enable verbose logging

**Example:**
```bash
python scripts/run_ingestion_pipeline.py data/input data/output --config high_quality --workers 8
```

#### `run_consolidation_pipeline.py`
Runs the knowledge consolidation pipeline.

**Features:**
- Document-level consolidation
- Subject-level consolidation
- Document clustering
- Knowledge storage
- Cluster-based consolidation

**Usage:**
```bash
python scripts/run_consolidation_pipeline.py input_path output_path [options]
```

**Options:**
- `--config` - Configuration preset (default, fast, high_quality)
- `--no-clustering` - Disable document clustering
- `--verbose` - Enable verbose logging

**Example:**
```bash
python scripts/run_consolidation_pipeline.py data/processed data/consolidated --config high_quality
```

### Database Management Scripts

#### `init_db.py`
Initializes the database schema and creates necessary tables.

**Features:**
- Database connection testing
- Table creation
- Index creation
- View creation
- Initial data insertion
- Setup verification

**Usage:**
```bash
python scripts/init_db.py [options]
```

**Options:**
- `--database-url` - Database connection URL
- `--drop-existing` - Drop existing tables
- `--verbose` - Enable verbose logging

**Example:**
```bash
python scripts/init_db.py --database-url postgresql://user:pass@localhost:5432/lerk_db
```

#### `migrate_db.py`
Handles database migrations and schema updates.

**Features:**
- Migration version tracking
- Schema updates
- Rollback support
- Migration status reporting

**Usage:**
```bash
python scripts/migrate_db.py [options]
```

**Options:**
- `--database-url` - Database connection URL
- `--migrate-to` - Target version to migrate to
- `--rollback` - Version to rollback to
- `--status` - Show migration status
- `--verbose` - Enable verbose logging

**Examples:**
```bash
# Show migration status
python scripts/migrate_db.py --status

# Migrate to latest version
python scripts/migrate_db.py

# Migrate to specific version
python scripts/migrate_db.py --migrate-to 1.4.0

# Rollback to specific version
python scripts/migrate_db.py --rollback 1.3.0
```

### Testing and Quality Scripts

#### `run_all_tests.py`
Runs comprehensive test suite for the LERK System.

**Features:**
- Module-specific testing
- Integration testing
- Performance testing
- Parallel test execution
- Comprehensive reporting
- JSON output support

**Usage:**
```bash
python scripts/run_all_tests.py [options]
```

**Options:**
- `--modules` - Specific modules to test
- `--verbose` - Enable verbose output
- `--coverage` - Enable coverage reporting
- `--integration` - Run integration tests
- `--performance` - Run performance tests
- `--output` - Output file for results
- `--json` - Output results in JSON format

**Examples:**
```bash
# Run all tests
python scripts/run_all_tests.py

# Run specific modules
python scripts/run_all_tests.py --modules ingest enrichment

# Run with coverage
python scripts/run_all_tests.py --coverage --verbose

# Run integration tests
python scripts/run_all_tests.py --integration --performance
```

#### `coverage_report.py`
Generates comprehensive code coverage reports.

**Features:**
- Module-specific coverage
- Combined coverage analysis
- HTML report generation
- Branch coverage analysis
- Coverage threshold checking
- Detailed reporting

**Usage:**
```bash
python scripts/coverage_report.py [options]
```

**Options:**
- `--modules` - Specific modules to analyze
- `--verbose` - Enable verbose output
- `--combined` - Run combined coverage analysis
- `--output` - Output file for results
- `--html` - Generate HTML coverage report
- `--html-dir` - Directory for HTML report
- `--json` - Output results in JSON format

**Examples:**
```bash
# Generate coverage report
python scripts/coverage_report.py

# Generate HTML report
python scripts/coverage_report.py --html --html-dir coverage_html

# Combined coverage analysis
python scripts/coverage_report.py --combined --verbose
```

## 🔧 Configuration

### Environment Variables

The scripts use the following environment variables:

```bash
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

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/lerk.log

# Processing Configuration
MAX_WORKERS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

### Configuration Files

- **`config/dev.yaml`** - Development configuration
- **`config/production.yaml`** - Production configuration
- **`config/test.yaml`** - Test configuration

## 📊 Monitoring and Logging

### Log Files

All scripts generate log files in the `logs/` directory:

- `logs/ingestion_pipeline.log` - Ingestion pipeline logs
- `logs/consolidation_pipeline.log` - Consolidation pipeline logs
- `logs/db_init.log` - Database initialization logs
- `logs/db_migration.log` - Database migration logs
- `logs/env_setup.log` - Environment setup logs
- `logs/dependency_check.log` - Dependency check logs
- `logs/test_runner.log` - Test runner logs
- `logs/coverage_report.log` - Coverage report logs

### Statistics and Reporting

Scripts provide comprehensive statistics and reporting:

- **Pipeline Statistics** - Processing times, throughput, success rates
- **Test Results** - Test counts, pass/fail rates, coverage percentages
- **Database Status** - Connection status, migration status, table counts
- **Dependency Status** - Package versions, system requirements, configuration

## 🚨 Troubleshooting

### Common Issues

1. **Dependency Installation Failures**
   ```bash
   # Check system dependencies
   python scripts/check_dependencies.py --verbose
   
   # Reinstall dependencies
   ./scripts/install_deps.sh
   ```

2. **Database Connection Issues**
   ```bash
   # Check database status
   python scripts/check_dependencies.py
   
   # Reinitialize database
   python scripts/init_db.py --drop-existing
   ```

3. **Test Failures**
   ```bash
   # Run tests with verbose output
   python scripts/run_all_tests.py --verbose
   
   # Check specific module
   python scripts/run_all_tests.py --modules ingest --verbose
   ```

4. **Pipeline Failures**
   ```bash
   # Check logs
   tail -f logs/ingestion_pipeline.log
   
   # Run with verbose output
   python scripts/run_ingestion_pipeline.py input output --verbose
   ```

### Performance Optimization

1. **Parallel Processing**
   ```bash
   # Increase worker count
   python scripts/run_ingestion_pipeline.py input output --workers 8
   ```

2. **Async Processing**
   ```bash
   # Enable async processing (default)
   python scripts/run_ingestion_pipeline.py input output
   ```

3. **Configuration Presets**
   ```bash
   # Use fast configuration
   python scripts/run_ingestion_pipeline.py input output --config fast
   
   # Use high quality configuration
   python scripts/run_ingestion_pipeline.py input output --config high_quality
   ```

## 📚 Additional Resources

- **Main Documentation** - See `README.md` in project root
- **Architecture Documentation** - See `docs/architecture.md`
- **Module Documentation** - See individual module README files
- **API Documentation** - See `services/api/README.md`

## 🤝 Contributing

When adding new scripts:

1. Follow the existing naming conventions
2. Include comprehensive help text and documentation
3. Add appropriate logging and error handling
4. Include usage examples
5. Update this README with new script documentation

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
