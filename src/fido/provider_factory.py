"""Provider construction."""

import threading
from pathlib import Path

import requests as _requests

from fido import provider as _provider
from fido.appstate import FidoState
from fido.atomic import AtomicUpdater
from fido.claude import (
    ClaudeAPI,
    ClaudeClient,
    ClaudeCode,
    ClaudeSessionFactory,
    ClaudeSessionFactoryMaker,
    StreamingRunner,
    _load_claude_oauth_state,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _RealClaudeSessionFactoryMaker,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _RealStreamingRunner,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from fido.codex import Codex, CodexAPI, CodexClient
from fido.config import RepoConfig
from fido.copilotcli import CopilotCLI, CopilotCLIAPI, CopilotCLIClient
from fido.infra import (
    Clock,
    IOSelector,
    PopenRunner,
    ProcessRunner,
    RealClock,
    RealIOSelector,
    RealPopenRunner,
    RealProcessRunner,
)
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
        claude_session_factory_maker: ClaudeSessionFactoryMaker,
        claude_streaming_runner: StreamingRunner,
        copilot_process_runner: ProcessRunner,
        copilot_popen_runner: PopenRunner,
    ) -> None:
        self._session_system_file = session_system_file
        self._claude_session_factory_maker = claude_session_factory_maker
        self._claude_streaming_runner = claude_streaming_runner
        self._copilot_process_runner = copilot_process_runner
        self._copilot_popen_runner = copilot_popen_runner
        self._api_lock = threading.Lock()
        self._apis: dict[ProviderID, ProviderAPI] = {}

    @classmethod
    def real(cls, *, session_system_file: Path) -> "DefaultProviderFactory":
        """Construct wired to real infrastructure — call from composition roots."""
        popen: PopenRunner = RealPopenRunner()
        selector: IOSelector = RealIOSelector()
        clock: Clock = RealClock()
        return cls(
            session_system_file=session_system_file,
            claude_session_factory_maker=_RealClaudeSessionFactoryMaker(
                popen=popen,
                selector=selector,
                clock=clock,
            ),
            claude_streaming_runner=_RealStreamingRunner(
                popen=popen,
                selector=selector,
                clock=clock,
            ),
            copilot_process_runner=RealProcessRunner(),
            copilot_popen_runner=popen,
        )

    def create_api(self, repo_cfg: RepoConfig) -> ProviderAPI:
        with self._api_lock:
            api = self._apis.get(repo_cfg.provider)
            if api is not None:
                return api
            match repo_cfg.provider:
                case ProviderID.CLAUDE_CODE:
                    api = ClaudeAPI(
                        session=_requests.Session(),
                        oauth_state_fn=_load_claude_oauth_state,
                        clock=RealClock(),
                    )
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
                claude_api = self.create_api(repo_cfg)
                assert isinstance(claude_api, ClaudeAPI)
                factory: ClaudeSessionFactory = self._claude_session_factory_maker(
                    work_dir=work_dir, repo_name=repo_name
                )
                return ClaudeCode(
                    api=claude_api,
                    agent=ClaudeClient(
                        streaming_runner=self._claude_streaming_runner,
                        session_factory=factory,
                        session_fn=_provider.current_repo_session,
                        session_system_file=self._session_system_file,
                        work_dir=work_dir,
                        repo_name=repo_name,
                        session=session,
                        state_updater=state_updater,
                    ),
                    session=None,
                )
            case ProviderID.COPILOT_CLI:
                shared_api = self.create_api(repo_cfg)
                assert isinstance(shared_api, CopilotCLIAPI)
                return CopilotCLI(
                    api=shared_api,
                    agent=CopilotCLIClient(
                        runner=self._copilot_process_runner,
                        popen=self._copilot_popen_runner,
                        session_fn=_provider.current_repo_session,
                        session_factory=None,
                        api=shared_api,
                        session_system_file=self._session_system_file,
                        work_dir=work_dir,
                        repo_name=repo_name,
                        session=session,
                        state_updater=state_updater,
                    ),
                    session=None,
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
