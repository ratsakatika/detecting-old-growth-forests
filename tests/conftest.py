"""Shared pytest fixtures and session-wide test hygiene.

Several unit tests spawn Python subprocesses (import-isolation and torch-free
probes). When the suite runs under ``pytest --cov``, pytest-cov exports
``COVERAGE_PROCESS_START`` / ``COV_CORE_*`` so subprocesses can be measured, and
the virtual environment's coverage ``.pth`` hooks start coverage in any child
that inherits those variables. Each such child writes a parallel data file
(``.coverage.<host>.pid<N>.<rand>``) that pytest-cov does not always combine
away, so they accumulate in the repository root on every run.

The project deliberately does not count subprocess coverage (``scripts`` are
omitted from the floor and ``utils`` is covered in-process), so the autouse
fixture below strips those triggers for the duration of every test and restores
them afterwards. In-process coverage is driven by the pytest-cov plugin, not by
these variables, so the coverage report is unaffected; only the stray data files
stop being created. This generalises the per-test stripping in
``test_run_nested_cv.py`` to the whole suite.
"""

import os

import pytest

# Variables that switch on coverage measurement in a freshly started process.
_COVERAGE_ENV_VARS: tuple[str, ...] = ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG")
# Prefixes for pytest-cov's subprocess-propagation variables (COV_CORE_SOURCE, ...).
_COVERAGE_ENV_PREFIXES: tuple[str, ...] = ("COV_CORE",)


@pytest.fixture(autouse=True)
def _no_subprocess_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop test-spawned subprocesses from emitting stray coverage data files."""
    for name in _COVERAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in [key for key in os.environ if key.startswith(_COVERAGE_ENV_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
