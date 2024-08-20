.PHONY: quality

export PYTHONPATH = src

check_dirs := tests src utils

quality:
	ruff format --config pyproject.toml $(check_dirs)
	ruff check --fix --respect-gitignore --config pyproject.toml $(check_dirs)
