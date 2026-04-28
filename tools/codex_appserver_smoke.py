#!/usr/bin/env python3
"""Exercise the real Codex app-server transport outside Fido runtime.

This is intentionally not part of CI: it talks to the local Codex install and
account. Run it when changing Codex app-server integration code:

    ./pyproject python tools/codex_appserver_smoke.py --turn
    ./pyproject python tools/codex_appserver_smoke.py --turn --interrupt
"""

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fido.codex import CodexAPI, CodexSession
from fido.provider import ProviderLimitSnapshot, ProviderModel


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return]
    return str(value)


def _print_json(label: str, value: object) -> None:
    print(f"\n== {label} ==")
    print(json.dumps(value, default=_json_default, indent=2, sort_keys=True))


def _snapshot_summary(snapshot: ProviderLimitSnapshot) -> dict[str, Any]:
    closest = snapshot.closest_to_exhaustion()
    return {
        "provider": snapshot.provider,
        "unavailable_reason": snapshot.unavailable_reason,
        "closest": closest,
        "windows": snapshot.windows,
    }


def _run_rate_limits() -> None:
    snapshot = CodexAPI().get_limit_snapshot()
    _print_json("rate limits", _snapshot_summary(snapshot))
    if snapshot.unavailable_reason is not None:
        raise RuntimeError(snapshot.unavailable_reason)


def _run_turn(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="fido-codex-smoke-") as tmp:
        system_file = Path(tmp) / "system.md"
        system_file.write_text(
            "You are validating Fido's Codex app-server integration. Be concise.\n"
        )
        session = CodexSession(
            system_file,
            work_dir=args.work_dir,
            model=ProviderModel(args.model, args.effort),
            repo_name=None,
        )
        try:
            _print_json(
                "thread",
                {
                    "pid": session.pid,
                    "session_id": session.session_id,
                    "alive": session.is_alive(),
                },
            )
            if args.interrupt:
                with session:
                    session.send(args.prompt)
                    time.sleep(args.interrupt_delay)
                    session.interrupt_active_turn()
                    result = session.consume_until_result()
                _print_json(
                    "interrupted turn",
                    {
                        "result": result,
                        "last_turn_cancelled": session.last_turn_cancelled,
                        "sent_count": session.sent_count,
                        "received_count": session.received_count,
                    },
                )
                if not session.last_turn_cancelled:
                    raise RuntimeError("turn was not reported as cancelled")
            else:
                result = session.prompt(args.prompt)
                _print_json(
                    "turn",
                    {
                        "result": result,
                        "last_turn_cancelled": session.last_turn_cancelled,
                        "sent_count": session.sent_count,
                        "received_count": session.received_count,
                    },
                )
                if not result:
                    raise RuntimeError("turn returned no assistant text")
        finally:
            session.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: fido codex smoke ok",
        help="Prompt used for the optional turn check.",
    )
    parser.add_argument(
        "--turn",
        action="store_true",
        help="Also start a thread and run a prompt turn.",
    )
    parser.add_argument(
        "--interrupt",
        action="store_true",
        help="Interrupt the turn after --interrupt-delay seconds.",
    )
    parser.add_argument("--interrupt-delay", type=float, default=1.0)
    args = parser.parse_args()

    try:
        _run_rate_limits()
        if args.turn or args.interrupt:
            _run_turn(args)
    except Exception as exc:
        print(f"codex app-server smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
