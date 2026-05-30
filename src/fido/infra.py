"""Infrastructure port protocols and real implementations.

Each port groups one low-level concern — process execution, time/sleep,
filesystem queries, or OS-level process control — so callers can accept
a single typed collaborator instead of a bag of raw stdlib callables.

Real implementations delegate directly to the stdlib with no added logic.
Tests inject fakes or mocks constructed at the call site.
"""

import dataclasses
import os
import select as _select_module
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn, Protocol

# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------


class ProcessRunner(Protocol):
    """Runs external processes.

    Wraps :func:`subprocess.run` so callers can be tested without spawning
    real subprocesses.
    """

    def run(
        self,
        cmd: Sequence[str],
        *,
        check: bool = True,
        **kwargs: Any,  # noqa: ANN401  # forwarded to subprocess.run
    ) -> subprocess.CompletedProcess[str]:
        """Execute *cmd*, forwarding kwargs to :func:`subprocess.run`.

        Defaults to ``check=True`` so non-zero exits raise; callers that want
        to handle a non-zero returncode themselves must pass ``check=False``
        explicitly.
        """
        ...


class RealProcessRunner:
    """Real :class:`ProcessRunner` that delegates to :func:`subprocess.run`."""

    def run(
        self,
        cmd: Sequence[str],
        *,
        check: bool = True,
        **kwargs: Any,  # noqa: ANN401  # forwarded to subprocess.run
    ) -> subprocess.CompletedProcess[str]:
        # Default check=True per the fail-fast subprocess policy in CLAUDE.md
        # (silent non-zero exits are the subprocess equivalent of catch-log-continue).
        # Callers that legitimately want to handle a non-zero returncode
        # themselves must pass check=False explicitly.
        return subprocess.run(cmd, check=check, **kwargs)


# ---------------------------------------------------------------------------
# Clock / time
# ---------------------------------------------------------------------------


class Clock(Protocol):
    """Time and sleep operations.

    Wraps :func:`time.sleep` and :func:`time.monotonic` so callers can be
    tested without real wall-clock delays.
    """

    def sleep(self, secs: float) -> None:
        """Pause execution for *secs* seconds."""
        ...

    def monotonic(self) -> float:
        """Return a monotonic clock value in fractional seconds."""
        ...


class RealClock:
    """Real :class:`Clock` that delegates to :mod:`time`."""

    def sleep(self, secs: float) -> None:
        time.sleep(secs)

    def monotonic(self) -> float:
        return time.monotonic()


# ---------------------------------------------------------------------------
# Filesystem queries
# ---------------------------------------------------------------------------


class Filesystem(Protocol):
    """Filesystem queries.

    Wraps :func:`shutil.which` and :meth:`pathlib.Path.is_dir` so callers
    can be tested against a fake filesystem.
    """

    def which(self, name: str) -> str | None:
        """Return the full path to *name* on PATH, or ``None`` if not found."""
        ...

    def is_dir(self, path: Path) -> bool:
        """Return ``True`` if *path* is an existing directory."""
        ...


class RealFilesystem:
    """Real :class:`Filesystem` that delegates to :mod:`shutil` and :mod:`pathlib`."""

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()


# ---------------------------------------------------------------------------
# OS-level process control
# ---------------------------------------------------------------------------


class OsProcess(Protocol):
    """OS-level process control.

    Wraps :func:`os.execvp`, :func:`os.chdir`, :func:`os._exit`, and
    :func:`signal.signal`
    so callers can be tested without replacing the running process or
    mutating global OS state.
    """

    def execvp(self, file: str, args: list[str]) -> None:
        """Replace the running process image — equivalent to :func:`os.execvp`."""
        ...

    def exit(self, code: int) -> NoReturn:
        """Exit the current process immediately — equivalent to :func:`os._exit`."""
        ...

    def chdir(self, path: Path | str) -> None:
        """Change the process working directory — equivalent to :func:`os.chdir`."""
        ...

    def install_signal(self, signum: int, handler: Callable[..., object]) -> object:
        """Install a signal handler — equivalent to :func:`signal.signal`.

        Returns the previous handler.
        """
        ...


class _OsBackend(Protocol):
    """Injectable OS operation backend for :class:`RealOsProcess`.

    Each method maps one-to-one to an OS or signal stdlib call.  The real
    implementation delegates directly; fakes record calls for tests.
    """

    def execvp(self, file: str, args: list[str]) -> None: ...

    def exit(self, code: int) -> NoReturn: ...

    def chdir(self, path: Path | str) -> None: ...

    def signal(self, signum: int, handler: Callable[..., object]) -> object: ...


class _StdlibOsBackend:  # pragma: no cover
    """Real :class:`_OsBackend` that delegates to :mod:`os` and :mod:`signal`.

    Not tested directly — these calls have irreversible process-level side
    effects (process replacement, immediate termination, cwd mutation,
    signal-table mutation).
    """

    def execvp(self, file: str, args: list[str]) -> None:
        os.execvp(file, args)

    def exit(self, code: int) -> NoReturn:
        os._exit(code)

    def chdir(self, path: Path | str) -> None:
        os.chdir(path)

    def signal(self, signum: int, handler: Callable[..., object]) -> object:
        return signal.signal(signum, handler)


_STDLIB_OS_BACKEND = _StdlibOsBackend()


class RealOsProcess:
    """Real :class:`OsProcess` that delegates to :class:`_OsBackend`.

    The default backend calls the stdlib directly.  Tests inject a fake
    :class:`_OsBackend` to record calls without running real OS operations.
    """

    def __init__(self, backend: _OsBackend = _STDLIB_OS_BACKEND) -> None:
        self._backend = backend

    def execvp(self, file: str, args: list[str]) -> None:
        self._backend.execvp(file, args)

    def exit(self, code: int) -> NoReturn:
        self._backend.exit(code)

    def chdir(self, path: Path | str) -> None:
        self._backend.chdir(path)

    def install_signal(self, signum: int, handler: Callable[..., object]) -> object:
        return self._backend.signal(signum, handler)


# ---------------------------------------------------------------------------
# Process spawning (long-lived subprocesses)
# ---------------------------------------------------------------------------


class PopenRunner(Protocol):
    """Spawns long-lived subprocesses.

    Wraps :class:`subprocess.Popen` so callers can be tested without spawning
    real processes.  Use :class:`ProcessRunner` for fire-and-forget
    ``subprocess.run`` calls; use :class:`PopenRunner` when you need to
    interact with the process while it runs (streaming output, sending signals,
    communicating via stdin/stdout, etc.).
    """

    def spawn(
        self,
        cmd: Sequence[str],
        **kwargs: Any,  # noqa: ANN401  # forwarded to subprocess.Popen
    ) -> "subprocess.Popen[str]":
        """Spawn *cmd* as a subprocess, forwarding kwargs to :class:`subprocess.Popen`.

        Returns the :class:`subprocess.Popen` instance immediately; the caller
        is responsible for waiting on it and reading its output streams.
        """
        ...


class RealPopenRunner:
    """Real :class:`PopenRunner` that delegates to :class:`subprocess.Popen`."""

    def spawn(
        self,
        cmd: Sequence[str],
        **kwargs: Any,  # noqa: ANN401  # forwarded to subprocess.Popen
    ) -> "subprocess.Popen[str]":
        return subprocess.Popen(cmd, **kwargs)


# ---------------------------------------------------------------------------
# I/O multiplexing
# ---------------------------------------------------------------------------


class IOSelector(Protocol):
    """I/O multiplexing over sets of file objects.

    Wraps :func:`select.select` so streaming read loops can be tested without
    blocking on real file descriptors.
    """

    def select(
        self,
        rlist: list[Any],
        wlist: list[Any],
        xlist: list[Any],
        timeout: float | None = None,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Wait until one or more file objects in the given lists are ready.

        Mirrors :func:`select.select`.  *timeout* ``None`` blocks indefinitely;
        ``0.0`` polls without blocking.  Returns a 3-tuple of subsets of the
        input lists that are ready for reading, writing, and exceptional
        conditions respectively.
        """
        ...


class RealIOSelector:
    """Real :class:`IOSelector` that delegates to :func:`select.select`."""

    def select(
        self,
        rlist: list[Any],
        wlist: list[Any],
        xlist: list[Any],
        timeout: float | None = None,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        return _select_module.select(rlist, wlist, xlist, timeout)


# ---------------------------------------------------------------------------
# Grouped bundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Infra:
    """All four infrastructure ports bundled as a single injectable collaborator.

    Callers that need all ports accept one :class:`Infra` instead of four
    separate arguments.  Tests construct an :class:`Infra` with fakes and
    inject the whole bundle at the composition root.
    """

    proc: ProcessRunner
    clock: Clock
    fs: Filesystem
    os_proc: OsProcess


def real_infra() -> Infra:
    """Construct an :class:`Infra` wired to the real stdlib implementations."""
    return Infra(
        proc=RealProcessRunner(),
        clock=RealClock(),
        fs=RealFilesystem(),
        os_proc=RealOsProcess(),
    )
