# Sportball Development Makefile
# Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
# Generated via Cursor IDE (cursor.sh) with AI assistance

.PHONY: help install install-dev test lint format clean setup

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
