"""
Repo-wide code-standard checks.

These enforce the project convention that every function and method in the
``unblend`` package is fully type-annotated, and that every one not nested
inside another function carries a reST-style docstring documenting each
parameter and any return value, one field per line. They are pure-AST checks:
fast, network-free, and safe to run in CI.
"""

import ast
import pathlib
import re

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "unblend"


_FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _iter_functions(
    include_nested: bool = True,
) -> list[tuple[pathlib.Path, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """
    Collect function/method definitions in the ``unblend`` package.

    :param include_nested: Also return functions defined inside functions.
    :return: List of ``(path, node)`` pairs.
    """
    found: list[tuple[pathlib.Path, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        nested: set[ast.AST] = set()
        for node in ast.walk(tree):
            if isinstance(node, _FUNCTION_TYPES):
                for child in ast.walk(node):
                    if child is not node and isinstance(child, _FUNCTION_TYPES):
                        nested.add(child)
        for node in ast.walk(tree):
            if isinstance(node, _FUNCTION_TYPES) and (
                include_nested or node not in nested
            ):
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
    Every non-nested function/method has a reST docstring covering its params
    and return, each field starting its own line with a real description.
    """
    problems: list[str] = []
    for path, node in _iter_functions(include_nested=False):
        where = f"{path.name}:{node.lineno} {node.name}"
        doc = ast.get_docstring(node)
        if not doc:
            problems.append(f"{where} -> no docstring")
            continue
        for name in _param_names(node):
            if not re.search(rf"^:param \*{{0,2}}{re.escape(name)}:", doc, re.M):
                problems.append(f"{where} -> missing ':param {name}:' line")
        if _returns_value(node) and not re.search(r"^:return", doc, re.M):
            problems.append(f"{where} -> missing ':return:' line")
        if re.search(r"\S[ \t]+:(param|return|raises)\b", doc):
            problems.append(f"{where} -> field list run together on one line")
        if re.search(
            r"^:param (\w+): \1 parameter\.$|^:return: Return value\.$", doc, re.M
        ):
            problems.append(f"{where} -> placeholder field description")
    assert not problems, "Docstring issues:\n" + "\n".join(problems)
