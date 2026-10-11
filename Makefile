PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: help install install-dev install-train lint format test paper clean

help:
	@echo "Targets: install install-dev install-train lint format test paper clean"

install:
	$(PIP) install -r requirements/base.txt
	$(PIP) install -e . --no-deps

install-dev: install
	$(PIP) install -r requirements/dev.txt
	pre-commit install

install-train: install
	$(PIP) install -r requirements/train.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	$(PYTHON) -m pytest

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	cd paper && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz
