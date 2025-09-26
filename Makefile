# Sportball Development Makefile
# Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
# Generated via Cursor IDE (cursor.sh) with AI assistance

.PHONY: help install install-dev test lint format clean setup build-rust-sidecar test-rust-sidecar bench-rust-sidecar benchmark-rust test-integration

# Default target
help:
	@echo "Sportball Development Commands:"
	@echo "  make install      - Install sportball package"
	@echo "  make install-dev  - Install sportball with development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make setup        - Complete setup (install + test)"
	@echo "  make build-rust-sidecar - Build Rust sidecar tool"
	@echo "  make test-rust-sidecar  - Test Rust sidecar tool"
	@echo "  make bench-rust-sidecar - Benchmark Rust sidecar tool"
	@echo "  make benchmark-rust     - Run Rust sidecar benchmark"
	@echo "  make test-integration   - Test Python-Rust integration"

# Install the package
install: venv
	${VENV}/bin/pip install -e .

# Install with development dependencies
install-dev: venv
	${VENV}/bin/pip install -e .[dev]

# Create virtual environment
venv:
	python -m venv venv
	${VENV}/bin/pip install --upgrade pip setuptools wheel

# Run tests
test: install-dev
	${VENV}/bin/pytest tests/ -v

# Run linting
lint: install-dev
	${VENV}/bin/flake8 sportball/
	${VENV}/bin/mypy sportball/

# Format code
format: install-dev
	${VENV}/bin/black sportball/
	${VENV}/bin/isort sportball/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Complete setup
setup: install-dev test
	@echo "✅ Sportball setup completed successfully!"

# Test CLI
test-cli: install
	${VENV}/bin/sportball --help
	${VENV}/bin/sb --help

# Run quick test
quick-test: install
	python test_sportball.py

# Install with CUDA support
install-cuda: venv
	${VENV}/bin/pip install -e .[cuda]

# Show package info
info: install
	${VENV}/bin/pip show sportball

# Uninstall
uninstall:
	${VENV}/bin/pip uninstall sportball -y

# Reinstall (clean + install)
reinstall: clean uninstall install

# Development mode (watch for changes)
dev: install-dev
	@echo "Development mode - watching for changes..."
	${VENV}/bin/watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='make test' \
		sportball/

# Build package
build: clean
	python -m build

# Install from built package
install-built: build
	${VENV}/bin/pip install dist/sportball-*.whl

# Show help for sportball CLI
cli-help: install
	${VENV}/bin/sportball --help
	@echo ""
	@echo "Available commands:"
	${VENV}/bin/sportball face --help
	${VENV}/bin/sportball object --help
	${VENV}/bin/sportball games --help
	${VENV}/bin/sportball ball --help
	${VENV}/bin/sportball quality --help
	${VENV}/bin/sportball util --help

# Rust sidecar tool targets
RUST_SIDECAR_DIR = ../sportball-sidecar-rust
RUST_SIDECAR_BINARY = $(RUST_SIDECAR_DIR)/target/release/sportball-sidecar-rust

# Build Rust sidecar tool
build-rust-sidecar:
	cd $(RUST_SIDECAR_DIR) && cargo build --release

# Test Rust sidecar tool
test-rust-sidecar: build-rust-sidecar
	cd $(RUST_SIDECAR_DIR) && cargo test

# Benchmark Rust sidecar tool
bench-rust-sidecar: build-rust-sidecar
	cd $(RUST_SIDECAR_DIR) && cargo bench

# Run Rust sidecar benchmark
benchmark-rust: install build-rust-sidecar
	${VENV}/bin/python -c "from sportball.detection.rust_sidecar import RustSidecarManager; import tempfile; import os; temp_dir = tempfile.mkdtemp(); [open(os.path.join(temp_dir, f'test_{i}.json'), 'w').write('{\"test\": \"data_' + str(i) + '\"}') for i in range(1000)]; manager = RustSidecarManager(); print('Rust available:', manager.rust_available); results = manager.validate_sidecars(temp_dir); print(f'Validated {len(results)} files'); import shutil; shutil.rmtree(temp_dir)"

# Test Python-Rust integration
test-integration: install build-rust-sidecar
	${VENV}/bin/python -c "from sportball.sidecar import Sidecar; from pathlib import Path; import tempfile; import os; temp_dir = Path(tempfile.mkdtemp()); [open(temp_dir / f'test_{i}.json', 'w').write('{\"test\": \"data_' + str(i) + '\"}') for i in range(100)]; sidecar = Sidecar(); print('Rust manager available:', sidecar.rust_manager.rust_available if sidecar.rust_manager else False); stats = sidecar.get_statistics(temp_dir); print(f'Statistics: {stats[\"total_sidecars\"]} sidecars'); import shutil; shutil.rmtree(temp_dir)"
