"""Shared pytest fixtures for fido tests."""

import faulthandler
import os
import signal
from typing import TextIO

import pytest

from fido import provider


def _open_faulthandler_log() -> TextIO:
    """Open a per-pid log file for SIGUSR1-driven Python stack dumps.

    Mirrors the production hook at :data:`fido.server` so any pytest worker
    can be inspected mid-hang via ``kill -SIGUSR1 <pid>`` (#1248).  The file
    is intentionally left open for the lifetime of the process — closing it
    would defeat the handler when it fires.
    """
    return open(f"/tmp/pyfh-{os.getpid()}.log", "w", encoding="utf-8", buffering=1)


# Each pytest process — controller and every xdist worker — registers its own
# SIGUSR1 handler at module import.  Importing conftest is the earliest hook
# pytest runs in a worker, well before any test collection or teardown.
_FAULTHANDLER_LOG = _open_faulthandler_log()
faulthandler.register(
    signal.SIGUSR1,
    file=_FAULTHANDLER_LOG,
    all_threads=True,
    chain=False,
)


@pytest.fixture(autouse=True)
def _reset_claude_talker_registry():
    """Clear the global :class:`~fido.provider.SessionTalker` registry between
    tests so entries from one test can't leak into the next and cause a
    spurious :class:`~fido.provider.SessionLeakError`.  Also clears any
    thread-local repo_name the test may have set via
    :func:`fido.provider.set_thread_repo`.
    """
    yield
    with provider._talkers_lock:
        provider._talkers.clear()
    provider.set_thread_repo(None)
