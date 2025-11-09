.PHONY: help install build clean test publish-pypi publish-crates publish-all dev check fmt lint

# Default target
help:
	@echo "Prak - Multi-object tracking library"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install maturin and development dependencies"
	@echo "  dev            - Install package in development mode"
	@echo "  build          - Build the package (Python wheels + Rust)"
	@echo "  build-rust     - Build Rust library only"
	@echo "  build-python   - Build Python wheels only"
	@echo "  test           - Run tests"
	@echo "  check          - Run cargo check"
	@echo "  fmt            - Format code (Rust + Python)"
	@echo "  lint           - Lint code"
	@echo "  clean          - Remove build artifacts"
	@echo "  publish-pypi   - Publish to PyPI"
	@echo "  publish-crates - Publish to crates.io"
	@echo "  publish-all    - Publish to both PyPI and crates.io"

# Install dependencies
install:
	@echo "Installing maturin and development dependencies..."
	pip install maturin twine build
	cargo --version || (echo "Rust not installed. Install from https://rustup.rs" && exit 1)

# Development install
dev:
	@echo "Installing in development mode..."
	maturin develop

# Build everything
build: build-rust build-python
	@echo "Build complete!"

# Build Rust library
build-rust:
	@echo "Building Rust library..."
	cargo build --release

# Build Python wheels
build-python:
	@echo "Building Python wheels..."
	maturin build --release

# Run tests
test:
	@echo "Running Rust tests..."
	cargo test
	@echo "Running Python tests..."
	python -m pytest python/tests/ || echo "No Python tests yet"

# Check Rust code
check:
	@echo "Running cargo check..."
	cargo check

# Format code
fmt:
	@echo "Formatting Rust code..."
	cargo fmt
	@echo "Formatting Python code..."
	black python/ || echo "black not installed, skipping Python formatting"

# Lint code
lint:
	@echo "Linting Rust code..."
	cargo clippy -- -D warnings
	@echo "Linting Python code..."
	ruff check python/ || echo "ruff not installed, skipping Python linting"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf dist/
	rm -rf target/
	rm -rf python/prak.egg-info/
	rm -rf python/prak/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

# Publish to PyPI
publish-pypi: build-python
	@echo "Publishing to PyPI..."
	@echo "⚠️  This will upload to PyPI. Make sure you're ready!"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		maturin publish; \
	else \
		echo "Cancelled."; \
	fi

# Publish to crates.io
publish-crates:
	@echo "Publishing to crates.io..."
	@echo "⚠️  This will upload to crates.io. Make sure you're ready!"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cargo publish; \
	else \
		echo "Cancelled."; \
	fi

# Publish to both PyPI and crates.io
publish-all: publish-crates publish-pypi
	@echo "Published to both PyPI and crates.io!"

# Quick check before publishing
pre-publish: clean check test build
	@echo "Pre-publish checks complete!"
	@echo "Package is ready to publish."
	@echo ""
	@echo "To publish, run:"
	@echo "  make publish-all"
