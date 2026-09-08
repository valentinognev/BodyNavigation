"""Scan Python/src/cadac for the quality-survey counters."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

_METRIC_KEYS = (
    "files",
    "loc",
    "code",
    "store_get",
    "store_set",
    "store_names",
    "look_up",
    "np_zeros",
    "np_array",
    "typed_defs",
    "untyped_defs",
    "empty_terminate",
    "empty_initialize",
    "skew_defs",
    "cadac_sign_defs",
)

_STORE_GET = "store.get("
_STORE_SET = "store.set("
_STORE_NAMES = "store.names("
_LOOK_UP = "look_up("
_NP_ZEROS = "np.zeros("
_NP_ARRAY = "np.array("

# Files whose text contains a matching def at line start after optional spaces.
_SKEW_DEF = re.compile(r"^[ \t]*def _skew|^[ \t]*def skew\(", re.MULTILINE)
_CADAC_SIGN_DEF = re.compile(
    r"^[ \t]*def _cadac_sign|^[ \t]*def cadac_sign", re.MULTILINE
)


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _code_line_count(text: str) -> int:
    """Non-blank, non-comment source lines (comment = stripped line starts with #)."""
    n = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            n += 1
    return n


def _is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.returns is not None:
        return True
    args = node.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if arg.annotation is not None:
            return True
    if args.vararg is not None and args.vararg.annotation is not None:
        return True
    if args.kwarg is not None and args.kwarg.annotation is not None:
        return True
    return False


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr):
        return False
    value = stmt.value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _empty_pass_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = node.body
    if body and _is_docstring_expr(body[0]):
        body = body[1:]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def _count_defs(tree: ast.AST) -> tuple[int, int, int, int]:
    typed = untyped = empty_term = empty_init = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_typed(node):
            typed += 1
        else:
            untyped += 1
        if node.name == "terminate" and _empty_pass_body(node):
            empty_term += 1
        elif node.name == "initialize" and _empty_pass_body(node):
            empty_init += 1
    return typed, untyped, empty_term, empty_init


def scan_cadac(root: Path) -> dict:
    """Return integer quality counters for every ``.py`` file under ``root``.

    ``loc`` is total lines. ``code`` is non-blank, non-comment source lines.
    """
    root = Path(root)
    files = _iter_py_files(root)
    data = {key: 0 for key in _METRIC_KEYS}
    data["files"] = len(files)
    for path in files:
        text = path.read_text(encoding="utf-8")
        data["loc"] += len(text.splitlines())
        data["code"] += _code_line_count(text)
        data["store_get"] += text.count(_STORE_GET)
        data["store_set"] += text.count(_STORE_SET)
        data["store_names"] += text.count(_STORE_NAMES)
        data["look_up"] += text.count(_LOOK_UP)
        data["np_zeros"] += text.count(_NP_ZEROS)
        data["np_array"] += text.count(_NP_ARRAY)
        if _SKEW_DEF.search(text):
            data["skew_defs"] += 1
        if _CADAC_SIGN_DEF.search(text):
            data["cadac_sign_defs"] += 1
        typed, untyped, empty_term, empty_init = _count_defs(ast.parse(text))
        data["typed_defs"] += typed
        data["untyped_defs"] += untyped
        data["empty_terminate"] += empty_term
        data["empty_initialize"] += empty_init
    return data


def write_metrics(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_metrics(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cadac_quality.metrics")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_metrics(args.out, scan_cadac(args.root))


if __name__ == "__main__":
    main()
