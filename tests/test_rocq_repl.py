# pyright: reportPrivateUsage=false

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import IO, Any, cast

import pytest

from fido import rocq_repl
from fido.rocq_repl import (
    LoadedModel,
    ModelLoader,
    OcamlReference,
    PythonEvaluator,
    ReferenceInvocation,
    RocqRepl,
    RocqReplError,
    ValueNormalizer,
)

REPO = Path(__file__).resolve().parents[1]


class _FakeRunner:
    """Typed :class:`~fido.infra.ProcessRunner` fake for ``OcamlReference`` tests.

    Either wraps a callable (*fn*) whose signature matches
    ``(cmd, **kwargs) -> SimpleNamespace``, or returns a fixed *return_value*
    from every call.  Recorded calls are available via :attr:`calls`.
    """

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        return_value: object = None,
    ) -> None:
        self._fn = fn
        self._return_value = return_value
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def run(
        self,
        cmd: Sequence[str],
        *,
        check: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(cmd), {"check": check, **kwargs}))
        if self._fn is not None:
            return self._fn(list(cmd), **kwargs)  # type: ignore[return-value]
        return self._return_value


def fake_module(**values: object) -> ModuleType:
    module = ModuleType("fake")
    for name, value in values.items():
        setattr(module, name, value)
    return module


def fake_import_module(_path: Path) -> ModuleType:
    return fake_module(toy=1)


def load_session_model() -> LoadedModel:
    return ModelLoader(REPO, StringIO()).load(Path("models/session_lock.v"))


def test_model_loader_binds_session_lock_symbols() -> None:
    model = load_session_model()

    assert "State" in model.namespace
    assert "Event" in model.namespace
    assert "transition" in model.namespace
    assert model.symbols["transition"].location().startswith("session_lock.v:")


def test_model_loader_falls_back_to_source_comments(tmp_path: Path) -> None:
    generated = tmp_path / "src" / "fido" / "rocq"
    generated.mkdir(parents=True)
    (tmp_path / "models").mkdir()
    source = tmp_path / "models" / "toy.v"
    source.write_text("Definition toy := 1.\n")
    module = generated / "toy.py"
    module.write_text("# From toy.v:1:0\n")

    model = ModelLoader(tmp_path, StringIO(), importer=fake_import_module).load(
        Path("models/toy.v")
    )

    assert model.namespace["toy"] == 1


def test_model_loader_warns_on_bad_map_and_uses_comment_fallback(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "src" / "fido" / "rocq"
    generated.mkdir(parents=True)
    (tmp_path / "models").mkdir()
    source = tmp_path / "models" / "toy.v"
    source.write_text("Definition toy := 1.\n")
    module = generated / "toy.py"
    module.write_text("# From toy.v:1:0\n")
    module.with_suffix(".pymap").write_text("{")
    err = StringIO()

    ModelLoader(tmp_path, err, importer=fake_import_module).load(Path("models/toy.v"))

    assert "could not parse" in err.getvalue()


def test_model_loader_errors_for_missing_source() -> None:
    with pytest.raises(RocqReplError, match="Rocq source file not found"):
        ModelLoader(REPO, StringIO()).load(Path("models/missing.v"))


def test_model_loader_errors_for_missing_generated_artifacts(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "src" / "fido" / "rocq").mkdir(parents=True)
    (tmp_path / "models" / "toy.v").write_text("Definition toy := 1.\n")

    with pytest.raises(RocqReplError, match="no extracted Python modules"):
        ModelLoader(tmp_path, StringIO()).load(Path("models/toy.v"))


def test_python_eval_and_normalizer() -> None:
    @dataclass
    class Box:
        value: int

    class ReprOnly:
        def __init__(self) -> None:
            self.value = 1

        def __repr__(self) -> str:
            return "repr-only"

    model = load_session_model()
    value = PythonEvaluator(model).evaluate("transition(Free(), WorkerAcquire())")

    assert ValueNormalizer().normalize(value) == "OwnedByWorker()"
    assert ValueNormalizer().normalize(None) == "None"
    assert ValueNormalizer().normalize(True) == "True"
    assert ValueNormalizer().normalize((1,)) == "(1,)"
    assert ValueNormalizer().normalize([None, 2]) == "[None, 2]"
    assert ValueNormalizer().normalize(Box(3)) == "Box(value=3)"
    assert ValueNormalizer().normalize(ReprOnly()) == "repr-only"


def test_reference_invocation_renders_extracted_rocq_symbol_call() -> None:
    model = load_session_model()
    invocation, python_value = ReferenceInvocation.from_expression(
        model, "transition(Free(), WorkerAcquire())"
    )

    assert ValueNormalizer().normalize(python_value) == "OwnedByWorker()"
    assert invocation.symbol == "transition"
    assert (
        invocation.ocaml_expression(model, "Session_lock_ref")
        == "(Session_lock_ref.transition Session_lock_ref.Free Session_lock_ref.WorkerAcquire)"
    )

    invocation, python_value = ReferenceInvocation.from_call(
        model,
        model.namespace["transition"],
        (
            cast(type, model.namespace["Free"])(),
            cast(type, model.namespace["WorkerAcquire"])(),
        ),
    )
    assert ValueNormalizer().normalize(python_value) == "OwnedByWorker()"
    assert invocation.symbol == "transition"


def test_reference_invocation_renders_rocq_values() -> None:
    class Box:
        __dataclass_fields__ = {"value": object()}

        def __init__(self, value: object) -> None:
            self.value = value

    model = LoadedModel(
        Path("x.v"),
        (),
        {"transition": object(), "Box": Box},
        {"transition": rocq_repl.RocqSymbol("transition", "x.v", 1, 0)},
    )
    invocation = ReferenceInvocation(
        expression="transition(...)",
        symbol="transition",
        args=(None, True, False, 1, "x", (1,), [2], Box(3)),
    )

    assert invocation.ocaml_expression(model, "M") == (
        '(M.transition M.None true false 1 "x" (1) [2] (M.Box 3))'
    )
    assert (
        ReferenceInvocation("transition", "transition", ()).ocaml_expression(model, "M")
        == "M.transition"
    )

    with pytest.raises(RocqReplError, match="non-Rocq value"):
        ReferenceInvocation(
            expression="transition(...)",
            symbol="transition",
            args=(object(),),
        ).ocaml_expression(model, "M")


def test_reference_invocation_rejects_unsupported_compare_shapes() -> None:
    model = load_session_model()

    with pytest.raises(RocqReplError, match="keyword arguments"):
        ReferenceInvocation.from_expression(model, "transition(current=Free())")
    with pytest.raises(RocqReplError, match="direct Rocq symbol call"):
        ReferenceInvocation.from_expression(model, "(lambda x: x)(1)")
    with pytest.raises(RocqReplError, match="direct Rocq symbol call"):
        ReferenceInvocation.from_expression(model, "1 + 2")
    with pytest.raises(RocqReplError, match="not bound"):
        ReferenceInvocation.from_expression(model, "missing()")

    with pytest.raises(RocqReplError, match="bound Rocq symbol"):
        ReferenceInvocation.from_call(model, object(), ())

    bad_model = LoadedModel(
        Path("x.v"),
        (),
        {"helper": object()},
        {},
    )
    with pytest.raises(RocqReplError, match="not an extracted Rocq symbol"):
        ReferenceInvocation.from_call(bad_model, "helper", ())

    noncall_model = LoadedModel(
        Path("x.v"),
        (),
        {"transition": object()},
        {"transition": rocq_repl.RocqSymbol("transition", "x.v", 1, 0)},
    )
    with pytest.raises(RocqReplError, match="not callable"):
        ReferenceInvocation.from_call(noncall_model, "transition", ())


def test_ocaml_reference_strips_python_pragmas_and_writes_eval(tmp_path: Path) -> None:
    model = load_session_model()
    ref = OcamlReference(REPO, model, StringIO())
    stripped = ref._strip_python_extraction(
        'A.\nExtract Inductive option => ""\n  "".\n'
        "Python Extraction transition.\n"
        "Python Module Extraction transition.\nB.\n"
    )
    assert "Extract Inductive option" not in stripped
    assert "Python Extraction" not in stripped
    assert "Python Module Extraction" not in stripped
    assert "A." in stripped
    assert "B." in stripped

    ref._write_eval(
        tmp_path,
        "Session_lock_ref",
        "Session_lock_ref.transition Session_lock_ref.Free",
    )
    eval_source = (tmp_path / "eval.ml").read_text()
    assert "normalize_state_option" in eval_source
    assert "OwnedByWorker()" in eval_source


def test_ocaml_reference_evaluate_runs_reference_toolchain(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path]] = []

    class SpyReference(OcamlReference):
        def _prepare_reference(self, work: Path) -> tuple[str, str]:
            calls.append((["prepare"], work))
            return "session_lock_ocaml_ref", "Session_lock_ocaml_ref"

        def _write_eval(self, work: Path, module_name: str, expression: str) -> None:
            calls.append((["write", module_name, expression], work))

        def _run_checked(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, argv: list[str], cwd: Path
        ) -> SimpleNamespace:
            calls.append((argv, cwd))
            return SimpleNamespace(returncode=0, stdout="OwnedByWorker()\n", stderr="")

    ref = SpyReference(REPO, load_session_model(), StringIO())
    invocation, _ = ReferenceInvocation.from_expression(
        load_session_model(), "transition(Free(), WorkerAcquire())"
    )
    assert ref.evaluate(invocation) == "OwnedByWorker()"
    assert calls[0][0] == ["prepare"]
    assert calls[-1][0] == ["./eval"]


def test_ocaml_reference_prepares_reference_project(tmp_path: Path) -> None:
    model = load_session_model()

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        assert argv == ["dune", "build", "models/session_lock_ocaml_ref.vo"]
        cwd = kwargs.get("cwd")
        assert isinstance(cwd, Path)
        generated = cwd / "_build" / "default" / "session_lock_ocaml_ref.ml"
        generated.parent.mkdir(parents=True)
        generated.write_text("let x = 1\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    ref = OcamlReference(REPO, model, StringIO(), runner=_FakeRunner(fn=fake_run))

    assert ref._prepare_reference(tmp_path) == (
        "session_lock_ocaml_ref",
        "Session_lock_ocaml_ref",
    )
    assert (tmp_path / "session_lock_ocaml_ref.ml").read_text() == "let x = 1\n"
    assert (
        "Extraction" in (tmp_path / "models" / "session_lock_ocaml_ref.v").read_text()
    )


def test_ocaml_reference_errors_when_reference_output_missing(tmp_path: Path) -> None:
    model = load_session_model()
    runner = _FakeRunner(
        return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    ref = OcamlReference(REPO, model, StringIO(), runner=runner)

    with pytest.raises(RocqReplError, match="did not produce"):
        ref._prepare_reference(tmp_path)


def test_ocaml_reference_raises_for_missing_state() -> None:
    model = LoadedModel(Path("x.v"), (), {"transition": object()}, {})
    ref = OcamlReference(REPO, model, StringIO())

    with pytest.raises(RocqReplError, match="requires a State"):
        ref._constructors()


def test_ocaml_reference_raises_for_empty_state_constructors() -> None:
    class State:
        pass

    model = LoadedModel(Path("x.v"), (), {"State": State}, {})
    ref = OcamlReference(REPO, model, StringIO())

    with pytest.raises(RocqReplError, match="found no State constructors"):
        ref._constructors()


def test_ocaml_reference_raises_when_command_fails(tmp_path: Path) -> None:
    failed = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    runner = _FakeRunner(return_value=failed)
    err = StringIO()
    ref = OcamlReference(REPO, load_session_model(), err, runner=runner)

    with pytest.raises(RocqReplError, match="command failed"):
        ref._run_checked(["false"], tmp_path)
    assert "boom" in err.getvalue()


def test_ocaml_reference_returns_successful_command_output(tmp_path: Path) -> None:
    ok = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
    runner = _FakeRunner(return_value=ok)
    ref = OcamlReference(REPO, load_session_model(), StringIO(), runner=runner)

    assert ref._run_checked(["true"], tmp_path).stdout == "ok\n"


def test_cli_eval_without_compare() -> None:
    stdout = StringIO()
    exit_code = RocqRepl(REPO, StringIO(), stdout, StringIO()).run(
        [
            "--no-compare",
            "--eval",
            "transition(Free(), WorkerAcquire())",
            "models/session_lock.v",
        ]
    )

    assert exit_code == 0
    assert stdout.getvalue() == "python: OwnedByWorker()\n"


def test_cli_eval_with_compare() -> None:
    class FakeReference:
        def __init__(
            self, repo_root: Path, model: LoadedModel, stderr: IO[str]
        ) -> None:
            self.repo_root = repo_root
            self.model = model
            self.stderr = stderr

        def evaluate(self, invocation: ReferenceInvocation) -> str:
            assert invocation.symbol == "transition"
            return "OwnedByWorker()"

    stdout = StringIO()

    exit_code = RocqRepl(
        REPO, StringIO(), stdout, StringIO(), reference_factory=FakeReference
    ).run(["--eval", "transition(Free(), WorkerAcquire())", "models/session_lock.v"])

    assert exit_code == 0
    assert "ocaml: OwnedByWorker()" in stdout.getvalue()
    assert "match: yes" in stdout.getvalue()


def test_compare_can_compute_python_result_from_invocation() -> None:
    class FakeReference:
        def __init__(
            self, repo_root: Path, model: LoadedModel, stderr: IO[str]
        ) -> None:
            self.repo_root = repo_root
            self.model = model
            self.stderr = stderr

        def evaluate(self, invocation: ReferenceInvocation) -> str:
            assert invocation.symbol == "transition"
            return "OwnedByWorker()"

    model = load_session_model()
    invocation, _ = ReferenceInvocation.from_expression(
        model, "transition(Free(), WorkerAcquire())"
    )

    result = RocqRepl(
        REPO, StringIO(), StringIO(), StringIO(), reference_factory=FakeReference
    )._compare(model, invocation)

    assert result.matches


def test_compare_rejects_noncallable_invocation_target() -> None:
    model = LoadedModel(
        Path("x.v"),
        (),
        {"transition": object()},
        {"transition": rocq_repl.RocqSymbol("transition", "x.v", 1, 0)},
    )
    invocation = ReferenceInvocation("transition(...)", "transition", ())

    with pytest.raises(RocqReplError, match="not callable"):
        RocqRepl(REPO, StringIO(), StringIO(), StringIO())._compare(model, invocation)


def test_cli_installs_helpers() -> None:
    class FakeReference:
        def __init__(
            self, repo_root: Path, model: LoadedModel, stderr: IO[str]
        ) -> None:
            self.repo_root = repo_root
            self.model = model
            self.stderr = stderr

        def evaluate(self, invocation: ReferenceInvocation) -> str:
            assert invocation.symbol == "transition"
            return "OwnedByWorker()"

    class FakeConsole:
        def __init__(self, locals: dict[str, object]) -> None:
            self.locals = locals

        def interact(self, banner: str, exitmsg: str) -> None:
            assert "rocq_symbols" in self.locals
            rocq_symbols = cast(Callable[[], list[str]], self.locals["rocq_symbols"])
            rocq_source = cast(Callable[[str], str], self.locals["rocq_source"])
            rocq_compare = cast(
                Callable[..., rocq_repl.CompareResult], self.locals["rocq_compare"]
            )
            assert "transition" in rocq_symbols()
            assert rocq_source("transition").startswith("session_lock.v:")
            with pytest.raises(KeyError):
                rocq_source("missing")
            transition = self.locals["transition"]
            free = cast(type, self.locals["Free"])
            worker_acquire = cast(type, self.locals["WorkerAcquire"])
            assert rocq_compare(transition, free(), worker_acquire()).matches
            assert "Bound symbols:" in banner
            assert exitmsg == ""

    assert (
        RocqRepl(
            REPO,
            StringIO(),
            StringIO(),
            StringIO(),
            reference_factory=FakeReference,
            console_factory=FakeConsole,
        ).run(["models/session_lock.v"])
        == 0
    )


def test_cli_reports_errors() -> None:
    err = StringIO()
    exit_code = RocqRepl(REPO, StringIO(), StringIO(), err).run(["models/missing.v"])

    assert exit_code == 1
    assert "error:" in err.getvalue()


def test_source_map_symbols_ignore_entries_without_symbol(tmp_path: Path) -> None:
    path = tmp_path / "x.pymap"
    path.write_text(
        "stability,python_start_line,python_start_col,python_end_line,"
        "python_end_col,source_file,source_start_line,source_start_col,"
        "source_end_line,source_end_col,kind,symbol\n"
        "open,1,0,1,0,x.v,1,0,1,0,extraction,\n"
    )

    assert ModelLoader(REPO, StringIO())._symbols_from_map(path) == {}


def test_main_uses_process_streams() -> None:
    assert rocq_repl.main(["models/missing.v"]) == 1
