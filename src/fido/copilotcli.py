"""Copilot CLI provider implementation."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from fido.acp import (
    ACPClient,
    ACPClientBase,
    ACPRuntime,
    ACPSession,
    combine_prompt,
    iter_jsonl,
    log_for_repo,
    transcript_block,
)
from fido.provider import (
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderAPI,
    ProviderID,
    ProviderLimitSnapshot,
    ProviderModel,
    coerce_provider_model,
)

log = logging.getLogger(__name__)

_COPILOT_COMMAND = ("copilot", "--acp", "--allow-all")
_COPILOT_JSON_BASE_ARGS = (
    "--output-format",
    "json",
    "--stream",
    "off",
    "--allow-all",
    "-s",
)

# #666: Copilot CLI emits this exact string as its final assistant content
# when a turn is cancelled mid-flight.
_COPILOT_CANCEL_SENTINEL = "Info: Operation cancelled by user"


def extract_result_text(output: str) -> str:
    """Extract the last assistant message content from Copilot JSONL output."""
    result = ""
    for obj in iter_jsonl(output):
        if obj.get("type") != "assistant.message":
            continue
        data = obj.get("data")
        if not isinstance(data, dict):
            continue
        content = data.get("content")
        if isinstance(content, str):
            result = content
    return result


def extract_session_id(output: str) -> str:
    """Extract the final Copilot session id from JSONL output."""
    result = ""
    for obj in iter_jsonl(output):
        if obj.get("type") != "result":
            continue
        session_id = obj.get("sessionId")
        if isinstance(session_id, str):
            result = session_id
    return result


def _normalize_model(model: ProviderModel | str | None) -> ProviderModel | None:
    if model is None:
        return None
    normalized = coerce_provider_model(model)
    lowered = normalized.model.lower()
    if lowered.startswith("claude-opus"):
        return ProviderModel("gpt-5.4", normalized.effort)
    if lowered.startswith("claude-sonnet"):
        return ProviderModel("gpt-5.4", normalized.effort)
    if lowered.startswith("claude-haiku"):
        return ProviderModel("gpt-5.4", normalized.effort)
    return normalized


def _copilot(
    *args: str,
    timeout: int = 30,
    cwd: Path | str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["copilot", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


class CopilotACPRuntime(ACPRuntime):
    """Own the long-lived Copilot ACP subprocess and connection."""

    def __init__(
        self,
        *,
        work_dir: Path,
        repo_name: str | None = None,
        command: Sequence[str] = _COPILOT_COMMAND,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ProviderID.COPILOT_CLI,
            command,
            work_dir=work_dir,
            repo_name=repo_name,
            normalize_model=_normalize_model,
            **kwargs,
        )

    def _default_client_factory(self, runtime: ACPRuntime) -> _CopilotACPClient:
        return _CopilotACPClient(runtime)


class _CopilotACPClient(ACPClientBase):
    """Copilot-specific ACP client implementation."""


class CopilotCLISession(ACPSession):
    """Persistent Copilot CLI ACP session."""

    def __init__(
        self,
        system_file: Path,
        *,
        work_dir: Path | str,
        model: ProviderModel | str,
        repo_name: str | None = None,
        runtime: CopilotACPRuntime | None = None,
        runtime_factory: Callable[..., CopilotACPRuntime] | None = None,
        session_id: str | None = None,
    ) -> None:
        if runtime is None:
            factory = CopilotACPRuntime if runtime_factory is None else runtime_factory
            runtime = factory(work_dir=Path(work_dir), repo_name=repo_name)
        super().__init__(
            system_file,
            work_dir=work_dir,
            model=model,
            cancel_sentinel=_COPILOT_CANCEL_SENTINEL,
            repo_name=repo_name,
            runtime=runtime,
            session_id=session_id,
        )


class CopilotCLIClient(ACPClient, ProviderAgent):
    """Injectable collaborator for Copilot CLI interactions."""

    voice_model = ProviderModel("gpt-5.4", "high")
    work_model = ProviderModel("gpt-5.4", "medium")
    brief_model = ProviderModel("gpt-5.4", "low")

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        session_factory: Callable[..., PromptSession] | None = None,
        **kwargs: Any,
    ) -> None:
        self._runner = runner
        self._session_factory = (
            CopilotCLISession if session_factory is None else session_factory
        )
        super().__init__(**kwargs)

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.COPILOT_CLI

    def _spawn_owned_session(
        self, model: ProviderModel, *, session_id: str | None = None
    ) -> PromptSession:
        system_file = self._session_system_file
        work_dir = self._work_dir
        assert system_file is not None
        assert work_dir is not None
        return self._session_factory(
            system_file,
            work_dir=work_dir,
            model=model,
            repo_name=self._repo_name,
            session_id=session_id,
        )

    def print_prompt_from_file(
        self,
        system_file: Path,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 30,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        del idle_timeout
        prompt = combine_prompt(
            prompt_file.read_text(),
            base_system_prompt=system_file.read_text(),
        )
        return self._run_cli_prompt(prompt, model=model, timeout=timeout, cwd=cwd)

    def resume_session(
        self,
        session_id: str,
        prompt_file: Path,
        model: ProviderModel,
        timeout: int = 300,
        idle_timeout: float = 1800.0,
        cwd: Path | str = ".",
    ) -> str:
        del idle_timeout
        return self._run_cli_prompt(
            prompt_file.read_text(),
            model=model,
            timeout=timeout,
            cwd=cwd,
            session_id=session_id,
        )

    def extract_session_id(self, output: str) -> str:
        return extract_session_id(output)

    def _run_cli_prompt(
        self,
        prompt: str,
        *,
        model: ProviderModel | str,
        timeout: int,
        cwd: Path | str | None = None,
        session_id: str | None = None,
    ) -> str:
        normalized = _normalize_model(model)
        assert normalized is not None
        log_for_repo(
            logging.INFO,
            self._repo_name,
            "%s",
            transcript_block("copilot prompt", prompt),
        )
        efforts = normalized.efforts or (None,)
        for effort in efforts:
            cmd = ["--model", normalized.model]
            if effort is not None:
                cmd += ["--effort", effort]
            cmd += [*_COPILOT_JSON_BASE_ARGS]
            if session_id is not None:
                cmd.append(f"--resume={session_id}")
            cmd += ["-p", prompt]
            result = _copilot(
                *cmd,
                timeout=timeout,
                cwd=self._work_dir if cwd is None else cwd,
                runner=self._runner,
            )
            if result.returncode == 0:
                text = extract_result_text(result.stdout.strip())
                log_for_repo(
                    logging.INFO,
                    self._repo_name,
                    "%s",
                    transcript_block("copilot result", text),
                )
                return result.stdout.strip()
        return ""


class CopilotCLI(Provider):
    """Composite Copilot provider with separate API and runtime agent."""

    def __init__(
        self,
        *,
        api: ProviderAPI | None = None,
        agent: ProviderAgent | None = None,
        session: PromptSession | None = None,
    ) -> None:
        if agent is None:
            agent = CopilotCLIClient(session=session)
        elif session is not None:
            agent.attach_session(session)
        self._api = CopilotCLIAPI() if api is None else api
        self._agent = agent

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.COPILOT_CLI

    @property
    def api(self) -> ProviderAPI:
        return self._api

    @property
    def agent(self) -> ProviderAgent:
        return self._agent


class CopilotCLIAPI(ProviderAPI):
    """Read-only account API for Copilot CLI."""

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.COPILOT_CLI

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        return ProviderLimitSnapshot(provider=self.provider_id)
