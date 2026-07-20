"""Regression checks for the isolated upstream benchmark worker."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _worker_template() -> str:
    """Read the upstream worker template without importing benchmark.py."""
    tree = ast.parse((ROOT / "benchmark.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_UPSTREAM_WORKER_TEMPLATE"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError("_UPSTREAM_WORKER_TEMPLATE not found")


def test_upstream_worker_imports_upstream_demucs() -> None:
    """The isolated worker must import the package installed in its venv."""
    template = _worker_template()
    assert "from demucs.api import Separator" in template
    assert "from unblend.api import Separator" not in template
    compile(template.replace("# __SHARED_SDR__", ""), "<upstream-worker>", "exec")
