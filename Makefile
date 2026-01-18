.PHONY: pybuild pytest test build

pybuild:
	rm -f python/multisensor_lmb_filters_rs/*.so
	rm -rf .venv/lib/python3.13/site-packages/multisensor_lmb_filters_rs*
	uv run --with pip pip list
	uv run --with pip maturin develop # --release
	uv run --with pip pip list

pytest: pybuild
	uv run pytest python/tests/ --disable-warnings -v

test:
	cargo test --release

build:
	cargo build --release

test-all: test pytest

build-all: build pybuild
