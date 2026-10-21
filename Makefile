.PHONY: install run test lint sandbox-image test-sandbox
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e '.[dev]'
run:
	$(PYTHON) -m streamlit run app.py
test:
	$(PYTHON) -m pytest -q -m 'not docker'
lint:
	$(PYTHON) -m ruff check src tests scripts app.py sandbox examples
	$(PYTHON) -m ruff format --check src tests scripts app.py sandbox examples
sandbox-image:
	docker build -f sandbox/Dockerfile -t pico-sandbox:1 .
test-sandbox:
	PICO_RUN_DOCKER_TESTS=1 $(PYTHON) -m pytest -q -m docker
