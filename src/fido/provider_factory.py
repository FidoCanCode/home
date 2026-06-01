"""Provider construction."""

import threading
from pathlib import Path

import requests as _requests

from fido.appstate import FidoState
from fido.atomic import AtomicUpdater
from fido.claude import (
    _REAL_SESSION_FACTORY_MAKER,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    ClaudeAPI,
    ClaudeClient,
    ClaudeCode,
    ClaudeSessionFactoryMaker,
)
from fido.codex import Codex, CodexAPI, CodexClient
from fido.config import RepoConfig
from fido.copilotcli import CopilotCLI, CopilotCLIAPI, CopilotCLIClient
from fido.infra import RealClock
from fido.provider import (
    PromptSession,
    Provider,
    ProviderAgent,
    ProviderAPI,
    ProviderID,
)


class DefaultProviderFactory:
    """Create repo-configured provider instances."""

    def __init__(
        self,
        *,
        session_system_file: Path,
        claude_session_factory_maker: ClaudeSessionFactoryMaker = _REAL_SESSION_FACTORY_MAKER,
    ) -> None:
        self._session_system_file = session_system_file
        self._claude_session_factory_maker = claude_session_factory_maker
        self._api_lock = threading.Lock()
        self._apis: dict[ProviderID, ProviderAPI] = {}

    def create_api(self, repo_cfg: RepoConfig) -> ProviderAPI:
        with self._api_lock:
            api = self._apis.get(repo_cfg.provider)
            if api is not None:
                return api
            match repo_cfg.provider:
                case ProviderID.CLAUDE_CODE:
                    api = ClaudeAPI(session=_requests.Session(), clock=RealClock())
                case ProviderID.COPILOT_CLI:
                    api = CopilotCLIAPI(clock=RealClock())
                case ProviderID.CODEX:
                    api = CodexAPI()
                case _:
                    raise ValueError(f"unsupported provider: {repo_cfg.provider}")
            self._apis[repo_cfg.provider] = api
            return api

    def create_provider(
        self,
        repo_cfg: RepoConfig,
        *,
        work_dir: Path,
        repo_name: str,
        session: PromptSession | None,
        state_updater: AtomicUpdater[FidoState] | None = None,
    ) -> Provider:
        match repo_cfg.provider:
            case ProviderID.CLAUDE_CODE:
                factory = self._claude_session_factory_maker(
                    work_dir=work_dir, repo_name=repo_name
                )
                return ClaudeCode(
                    agent=ClaudeClient(
                        session_system_file=self._session_system_file,
                        work_dir=work_dir,
                        repo_name=repo_name,
                        session=session,
                        state_updater=state_updater,
                        session_factory=factory,
                    )
                )
            case ProviderID.COPILOT_CLI:
                shared_api = self.create_api(repo_cfg)
                assert isinstance(shared_api, CopilotCLIAPI)
                return CopilotCLI(
                    api=shared_api,
                    agent=CopilotCLIClient(
                        api=shared_api,
                        session_system_file=self._session_system_file,
                        work_dir=work_dir,
                        repo_name=repo_name,
                        session=session,
                        state_updater=state_updater,
                    ),
                )
            case ProviderID.CODEX:
                return Codex(
                    agent=CodexClient(
                        session_system_file=self._session_system_file,
                        work_dir=work_dir,
                        repo_name=repo_name,
                        session=session,
                        state_updater=state_updater,
                    )
                )
            case _:
                raise ValueError(f"unsupported provider: {repo_cfg.provider}")

    def create_agent(
        self,
        repo_cfg: RepoConfig,
        *,
        work_dir: Path,
        repo_name: str,
    ) -> ProviderAgent:
        return self.create_provider(
            repo_cfg,
            work_dir=work_dir,
            repo_name=repo_name,
            session=None,
        ).agent
