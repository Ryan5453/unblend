"""
Repo-wide code-standard checks.

These enforce the project convention that *every* function and method in the
``demucs`` package is fully type-annotated and carries a reST-style docstring
that documents each parameter and any return value. They are pure-AST checks:
fast, network-free, and safe to run in CI.
"""

import ast
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "demucs"


def _iter_functions() -> list[
    tuple[pathlib.Path, ast.FunctionDef | ast.AsyncFunctionDef]
]:
    """
    Collect every function/method definition in the ``demucs`` package.

    :return: List of ``(path, node)`` pairs, including nested functions.
    """
    found: list[tuple[pathlib.Path, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append((path, node))
    return found


def _param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """
    Return the documentable parameter names of a function (excluding self/cls).

    :param node: Function definition to inspect.
    :return: Parameter names, with ``*args``/``**kwargs`` reported bare.
    """
    a = node.args
    names = [
        p.arg
        for p in (a.posonlyargs + a.args + a.kwonlyargs)
        if p.arg not in ("self", "cls")
    ]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _returns_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """
    Whether the function's return annotation is something other than ``None``.

    :param node: Function definition to inspect.
    :return: ``True`` if the annotated return type is not ``None``.
    """
    ret = node.returns
    if ret is None:
        return False
    return not (isinstance(ret, ast.Constant) and ret.value is None)


def test_all_functions_fully_typed() -> None:
    """
    Every function/method annotates all parameters and its return type.
    """
    problems: list[str] = []
    for path, node in _iter_functions():
        a = node.args
        missing = [
            p.arg
            for p in (a.posonlyargs + a.args + a.kwonlyargs)
            if p.arg not in ("self", "cls") and p.annotation is None
        ]
        if a.vararg and a.vararg.annotation is None:
            missing.append("*" + a.vararg.arg)
        if a.kwarg and a.kwarg.annotation is None:
            missing.append("**" + a.kwarg.arg)
        if node.returns is None:
            missing.append("<return>")
        if missing:
            problems.append(
                f"{path.name}:{node.lineno} {node.name} -> {', '.join(missing)}"
            )
    assert not problems, "Functions missing type annotations:\n" + "\n".join(problems)


def test_all_functions_have_rest_docstrings() -> None:
    """
    Every function/method has a reST docstring covering its params and return.
    """
    problems: list[str] = []
    for path, node in _iter_functions():
        doc = ast.get_docstring(node)
        if not doc:
            problems.append(f"{path.name}:{node.lineno} {node.name} -> no docstring")
            continue
        for name in _param_names(node):
            if not any(
                marker in doc
                for marker in (
                    f":param {name}:",
                    f":param *{name}:",
                    f":param **{name}:",
                )
            ):
                problems.append(
                    f"{path.name}:{node.lineno} {node.name} -> missing ':param {name}:'"
                )
        if _returns_value(node) and ":return" not in doc:
            problems.append(
                f"{path.name}:{node.lineno} {node.name} -> missing ':return:'"
            )
    assert not problems, "Docstring issues:\n" + "\n".join(problems)
