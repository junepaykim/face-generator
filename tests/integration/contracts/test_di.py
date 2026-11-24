# AGENT INVARIANT: Do not weaken or bypass these assertions to make implementation changes pass without explicit user approval.
"""Static dependency boundaries, including imports inside functions and TYPE_CHECKING."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path

from support.paths import ROOT

SOURCE = ROOT / "src" / "kitshot"
LAUNCHERS = {
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "asyncio.subprocess.create_subprocess_exec",
    "asyncio.subprocess.create_subprocess_shell",
    "os.system",
    "os.popen",
    "os.startfile",
    *(
        f"os.{kind}{suffix}"
        for kind in ("exec", "spawn")
        for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")
    ),
    "os.posix_spawn",
    "os.posix_spawnp",
}


def import_base(node: ast.ImportFrom, path: Path) -> str:
    if not node.level:
        return node.module or ""
    package = ".".join(path.parent.relative_to(ROOT / "src").parts)
    return resolve_name("." * node.level + (node.module or ""), package)


def scope_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Visit one lexical scope; nested functions/classes supply their own bindings."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from scope_nodes(child)


def qualified(node: ast.expr, bindings: dict[str, str | None]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        base = qualified(node.value, bindings)
        return f"{base}.{node.attr}" if base else None
    return None


def launch_calls(
    node: ast.AST, path: Path, inherited: dict[str, str | None]
) -> Iterator[tuple[int, str]]:
    bindings = inherited.copy()
    children = list(scope_nodes(node))
    # A local argument/assignment shadows an imported name from its enclosing scope.
    for child in children:
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            bindings[child.id] = None
        elif isinstance(child, ast.arg):
            bindings[child.arg] = None
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[child.name] = None
    for child in children:
        if isinstance(child, ast.Import):
            for alias in child.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(child, ast.ImportFrom):
            base = import_base(child, path)
            for alias in child.names:
                bindings[alias.asname or alias.name] = f"{base}.{alias.name}"
    for child in children:
        if isinstance(child, ast.Call):
            name = qualified(child.func, bindings)
            if name in LAUNCHERS:
                yield child.lineno, name
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from launch_calls(child, path, bindings)


def test_core_imports_only_core_or_the_standard_library() -> None:
    violations = []
    for path in sorted((SOURCE / "core").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = import_base(node, path)
                names = (
                    [f"{base}.{alias.name}" for alias in node.names]
                    if base == "kitshot"
                    else [base]
                )
            else:
                continue
            for name in names:
                if name == "kitshot.core" or name.startswith("kitshot.core."):
                    continue
                if name.split(".")[0] in sys.stdlib_module_names:
                    continue
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: imports {name}")
    assert not violations, (
        "Core must not depend on adapters or third-party packages:\n" + "\n".join(violations)
    )


def test_process_launches_go_through_the_quiet_platform_adapter() -> None:
    violations = []
    for path in sorted(SOURCE.rglob("*.py")):
        if path == SOURCE / "platform" / "procs.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, name in launch_calls(tree, path, {}):
            violations.append(f"{path.relative_to(ROOT)}:{line}: calls {name}")
    assert not violations, "Launch through procs.run_quiet/popen_quiet:\n" + "\n".join(violations)
