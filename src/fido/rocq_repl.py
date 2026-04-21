"""Interactive REPL for Rocq-extracted Python models."""

import argparse
import ast
import code
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import IO, Any

_IMPORT_MODULES = {
    "dataclasses",
    "itertools",
    "typing",
}
_IMPORT_NAMES = {
    "Any",
    "Awaitable",
    "Callable",
    "Generic",
    "Iterable",
    "Iterator",
    "Never",
    "Protocol",
    "TypeVar",
    "assert_never",
    "cast",
    "dataclass",
    "islice",
}


@dataclass(frozen=True)
class RocqSymbol:
    name: str
    source_file: str
    source_start_line: int
    source_start_col: int

    def location(self) -> str:
        return f"{self.source_file}:{self.source_start_line}:{self.source_start_col}"


@dataclass(frozen=True)
class LoadedModel:
    source: Path
    modules: tuple[ModuleType, ...]
    namespace: dict[str, object]
    symbols: dict[str, RocqSymbol]


@dataclass(frozen=True)
class CompareResult:
    expression: str
    python_result: str
    ocaml_result: str

    @property
    def matches(self) -> bool:
        return self.python_result == self.ocaml_result


class RocqReplError(RuntimeError):
    pass


class ModelLoader:
    def __init__(self, repo_root: Path, stderr: IO[str]) -> None:
        self._repo_root = repo_root
        self._stderr = stderr
        self._generated_dir = repo_root / "src" / "fido" / "rocq"

    def load(self, source: Path) -> LoadedModel:
        resolved = self._resolve_source(source)
        module_paths = self._module_paths_for_source(resolved)
        if not module_paths:
            raise RocqReplError(
                f"no extracted Python modules found for {source}; run ./fido make-rocq"
            )

        namespace: dict[str, object] = {}
        symbols: dict[str, RocqSymbol] = {}
        modules = tuple(self._import_module(path) for path in module_paths)
        for module, path in zip(modules, module_paths, strict=True):
            namespace.update(self._public_symbols(module))
            symbols.update(self._symbols_from_map(path.with_suffix(".pymap")))

        return LoadedModel(
            source=resolved,
            modules=modules,
            namespace=namespace,
            symbols=symbols,
        )

    def _resolve_source(self, source: Path) -> Path:
        candidate = source if source.is_absolute() else self._repo_root / source
        if not candidate.is_file():
            raise RocqReplError(f"Rocq source file not found: {source}")
        return candidate.resolve()

    def _module_paths_for_source(self, source: Path) -> tuple[Path, ...]:
        matches = [
            path
            for path in sorted(self._generated_dir.glob("*.py"))
            if path.name != "__init__.py" and self._module_matches_source(path, source)
        ]
        return tuple(matches)

    def _module_matches_source(self, path: Path, source: Path) -> bool:
        map_path = path.with_suffix(".pymap")
        if map_path.is_file():
            try:
                data = json.loads(map_path.read_text())
            except json.JSONDecodeError as exc:
                print(f"warning: could not parse {map_path}: {exc}", file=self._stderr)
            else:
                return any(
                    Path(str(entry.get("source_file", ""))).name == source.name
                    for entry in data.get("entries", [])
                    if isinstance(entry, dict)
                )

        marker = f"# From {source.name}:"
        return marker in path.read_text()

    def _import_module(self, path: Path) -> ModuleType:
        module_name = f"fido.rocq.{path.stem}"
        return importlib.import_module(module_name)

    def _public_symbols(self, module: ModuleType) -> dict[str, object]:
        public: dict[str, object] = {}
        for name, value in vars(module).items():
            if name.startswith("_") or name in _IMPORT_NAMES:
                continue
            value_module = getattr(value, "__module__", module.__name__)
            if value_module in _IMPORT_MODULES:
                continue
            public[name] = value
        return public

    def _symbols_from_map(self, path: Path) -> dict[str, RocqSymbol]:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"warning: could not parse {path}: {exc}", file=self._stderr)
            return {}
        symbols: dict[str, RocqSymbol] = {}
        for entry in data.get("entries", []):
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol", ""))
            if not symbol:
                continue
            symbols[symbol] = RocqSymbol(
                name=symbol,
                source_file=str(entry["source_file"]),
                source_start_line=int(entry["source_start_line"]),
                source_start_col=int(entry["source_start_col"]),
            )
        return symbols


class PythonEvaluator:
    def __init__(self, model: LoadedModel) -> None:
        self._model = model

    def evaluate(self, expression: str) -> object:
        parsed = ast.parse(expression, mode="eval")
        return eval(
            compile(parsed, "<rocq-python-repl>", "eval"), {}, self._model.namespace
        )


class ValueNormalizer:
    def normalize(self, value: object) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool | int | float | str):
            return repr(value)
        if isinstance(value, tuple):
            inner = ", ".join(self.normalize(item) for item in value)
            suffix = "," if len(value) == 1 else ""
            return f"({inner}{suffix})"
        if isinstance(value, list):
            return "[" + ", ".join(self.normalize(item) for item in value) + "]"
        name = type(value).__name__
        fields = getattr(value, "__dataclass_fields__", {})
        if fields:
            args = ", ".join(
                f"{field}={self.normalize(getattr(value, field))}" for field in fields
            )
            return f"{name}({args})"
        if hasattr(value, "__dict__") and not vars(value):
            return f"{name}()"
        return repr(value)


class PythonToOcaml:
    def __init__(self, model: LoadedModel) -> None:
        self._model = model

    def translate(self, expression: str, module_name: str) -> str:
        parsed = ast.parse(expression, mode="eval")
        return self._expr(parsed.body, module_name)

    def _expr(self, node: ast.AST, module_name: str) -> str:
        if isinstance(node, ast.Name):
            self._require_bound(node.id)
            return f"{module_name}.{node.id}"
        if isinstance(node, ast.Constant):
            return self._constant(node.value)
        if isinstance(node, ast.Tuple):
            return (
                "("
                + ", ".join(self._expr(item, module_name) for item in node.elts)
                + ")"
            )
        if isinstance(node, ast.List):
            return (
                "["
                + "; ".join(self._expr(item, module_name) for item in node.elts)
                + "]"
            )
        if isinstance(node, ast.Call):
            if node.keywords:
                raise RocqReplError("OCaml compare does not support keyword arguments")
            if not isinstance(node.func, ast.Name):
                raise RocqReplError("OCaml compare only supports direct name calls")
            self._require_bound(node.func.id)
            target = f"{module_name}.{node.func.id}"
            if not node.args:
                return target
            args = " ".join(self._expr(arg, module_name) for arg in node.args)
            return f"({target} {args})"
        raise RocqReplError(
            f"OCaml compare does not support {type(node).__name__} expressions"
        )

    def _constant(self, value: object) -> str:
        if value is None:
            return "None"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return json.dumps(value)
        raise RocqReplError(f"OCaml compare does not support literal {value!r}")

    def _require_bound(self, name: str) -> None:
        if name not in self._model.namespace:
            raise RocqReplError(f"name is not bound in the Rocq Python REPL: {name}")


class OcamlReference:
    def __init__(
        self,
        repo_root: Path,
        model: LoadedModel,
        stderr: IO[str],
        run: Any = subprocess.run,
    ) -> None:
        self._repo_root = repo_root
        self._model = model
        self._stderr = stderr
        self._run = run
        self._translator = PythonToOcaml(model)

    def evaluate(self, expression: str) -> str:
        with tempfile.TemporaryDirectory(prefix="rocq-repl-") as raw:
            work = Path(raw)
            source_name, module_name = self._prepare_reference(work)
            ocaml_expr = self._translator.translate(expression, module_name)
            self._write_eval(work, module_name, ocaml_expr)
            self._run_checked(
                ["ocamlc", "-o", "eval", f"{source_name}.ml", "eval.ml"], work
            )
            result = self._run_checked(["./eval"], work)
            return result.stdout.strip()

    def _prepare_reference(self, work: Path) -> tuple[str, str]:
        self._copy_project_files(work)
        source_name = self._model.source.stem + "_ocaml_ref"
        model_dir = work / "models"
        reference_source = model_dir / f"{source_name}.v"
        reference_source.write_text(self._reference_source())
        (model_dir / "dune").write_text(
            "(rocq.theory\n"
            " (name FidoModels)\n"
            ' (synopsis "Fido coordination model OCaml reference")\n'
            " (plugins rocq-python-extraction)\n"
            f" (modules {source_name}))\n"
        )
        self._run_checked(["dune", "build", f"models/{source_name}.vo"], work)
        generated = work / "_build" / "default" / f"{source_name}.ml"
        if not generated.is_file():
            raise RocqReplError(
                f"OCaml reference extraction did not produce {generated}"
            )
        shutil.copy(generated, work / f"{source_name}.ml")
        return source_name, source_name.capitalize()

    def _copy_project_files(self, work: Path) -> None:
        shutil.copy(self._repo_root / "dune-workspace", work / "dune-workspace")
        model_dir = work / "models"
        model_dir.mkdir()
        shutil.copy(
            self._repo_root / "models" / "dune-project", model_dir / "dune-project"
        )
        plugin_dir = work / "rocq-python-extraction"
        plugin_dir.mkdir()
        for name in (
            "dune-project",
            "dune",
            "rocq-python-extraction.opam",
            "META.rocq-python-extraction.template",
            "g_python_extraction.mlg",
            "python.ml",
        ):
            shutil.copy(
                self._repo_root / "rocq-python-extraction" / name, plugin_dir / name
            )

    def _reference_source(self) -> str:
        source = self._strip_python_extraction(self._model.source.read_text())
        extraction_commands = "\n".join(
            f'Extraction "{self._model.source.stem}_ocaml_ref.ml" {symbol}.'
            for symbol in sorted(self._model.symbols)
        )
        return f"{source.rstrip()}\n\n{extraction_commands}\n"

    def _strip_python_extraction(self, text: str) -> str:
        lines = text.splitlines()
        kept: list[str] = []
        skip_until_dot = False
        for line in lines:
            stripped = line.strip()
            if skip_until_dot:
                if stripped.endswith("."):
                    skip_until_dot = False
                continue
            if stripped.startswith("Python Extraction "):
                continue
            if stripped.startswith("Python Module Extraction "):
                continue
            if stripped.startswith("Extract Inductive option =>"):
                if not stripped.endswith("."):
                    skip_until_dot = True
                continue
            kept.append(line)
        return "\n".join(kept)

    def _write_eval(self, work: Path, module_name: str, expression: str) -> None:
        constructors = self._constructors()
        state_cases = "\n".join(
            f'  | {module_name}.{name} -> "{name}()"' for name in constructors
        )
        option_cases = "\n".join(
            f'  | {module_name}.Some {module_name}.{name} -> "{name}()"'
            for name in constructors
        )
        eval_source = (
            f"let normalize_state = function\n{state_cases}\n\n"
            f"let normalize_state_option = function\n"
            f'  | {module_name}.None -> "None"\n{option_cases}\n\n'
            f"let () = print_endline (normalize_state_option ({expression}))\n"
        )
        (work / "eval.ml").write_text(eval_source)

    def _constructors(self) -> tuple[str, ...]:
        base = self._model.namespace.get("State")
        if not isinstance(base, type):
            raise RocqReplError("OCaml compare currently requires a State inductive")
        constructors = [
            name
            for name, value in self._model.namespace.items()
            if isinstance(value, type) and issubclass(value, base) and value is not base
        ]
        if not constructors:
            raise RocqReplError("OCaml compare found no State constructors")
        return tuple(sorted(constructors))

    def _run_checked(
        self, argv: list[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PATH"] = f"/home/opam/.opam/5.3/bin:{env.get('PATH', '')}"
        result = self._run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            self._stderr.write(result.stderr)
            raise RocqReplError(f"command failed: {' '.join(argv)}")
        return result


class RocqRepl:
    def __init__(
        self,
        repo_root: Path,
        stdin: IO[str],
        stdout: IO[str],
        stderr: IO[str],
    ) -> None:
        self._repo_root = repo_root
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self._normalizer = ValueNormalizer()

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(
            prog="rocq-python-repl",
            description="Open a Python REPL preloaded with Rocq-extracted symbols.",
        )
        parser.add_argument("source")
        parser.add_argument("--eval", dest="expression")
        parser.add_argument("--no-compare", action="store_true")
        args = parser.parse_args(argv)

        try:
            model = ModelLoader(self._repo_root, self._stderr).load(Path(args.source))
            self._install_helpers(model)
            if args.expression is not None:
                self._run_eval(model, args.expression, compare=not args.no_compare)
            else:
                self._interact(model)
        except RocqReplError as exc:
            print(f"error: {exc}", file=self._stderr)
            return 1
        return 0

    def _install_helpers(self, model: LoadedModel) -> None:
        def rocq_symbols() -> list[str]:
            return sorted(model.namespace)

        def rocq_source(name: str) -> str:
            symbol = model.symbols.get(name)
            if symbol is None:
                raise KeyError(name)
            return symbol.location()

        def rocq_compare(expression: str) -> CompareResult:
            return self._compare(model, expression)

        model.namespace["rocq_symbols"] = rocq_symbols
        model.namespace["rocq_source"] = rocq_source
        model.namespace["rocq_compare"] = rocq_compare

    def _run_eval(self, model: LoadedModel, expression: str, *, compare: bool) -> None:
        python_value = PythonEvaluator(model).evaluate(expression)
        python_result = self._normalizer.normalize(python_value)
        self._stdout.write(f"python: {python_result}\n")
        if compare:
            result = self._compare(model, expression, python_result=python_result)
            self._stdout.write(f"ocaml: {result.ocaml_result}\n")
            self._stdout.write(f"match: {'yes' if result.matches else 'no'}\n")

    def _compare(
        self,
        model: LoadedModel,
        expression: str,
        *,
        python_result: str | None = None,
    ) -> CompareResult:
        if python_result is None:
            python_result = self._normalizer.normalize(
                PythonEvaluator(model).evaluate(expression)
            )
        ocaml_result = OcamlReference(self._repo_root, model, self._stderr).evaluate(
            expression
        )
        return CompareResult(
            expression=expression,
            python_result=python_result,
            ocaml_result=ocaml_result,
        )

    def _interact(self, model: LoadedModel) -> None:
        banner = (
            f"Rocq Python REPL for {model.source.relative_to(self._repo_root)}\n"
            f"Bound symbols: {', '.join(sorted(model.namespace))}\n"
            "Helpers: rocq_symbols(), rocq_source(name), rocq_compare(expr)"
        )
        console = code.InteractiveConsole(locals=model.namespace)
        console.interact(banner=banner, exitmsg="")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    return RocqRepl(repo_root(), sys.stdin, sys.stdout, sys.stderr).run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
