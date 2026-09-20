.PHONY: install lint test smoke ci clean

install:
	pip install -r requirements-lock.txt
	pip install --no-deps -e .
	pre-commit install

lint:
	pre-commit run --all-files

test:
	pytest -ra --strict-markers -m "not slow and not requires_data and not requires_gpu"

smoke:
	pytest -ra --strict-markers -m "not requires_data and not requires_gpu" -x

ci:
	pre-commit run --all-files
	pytest -ra --strict-markers \
		-m "not slow and not requires_data and not requires_gpu" \
		--cov=utils --cov=scripts --cov-report=term-missing \
		--cov-fail-under=98
	mypy utils $$([ -n "$$(find scripts -name '*.py' -print -quit 2>/dev/null)" ] && echo scripts)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
