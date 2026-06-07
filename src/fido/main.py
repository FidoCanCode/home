"""Top-level fido server entry point."""


def main() -> None:  # pragma: no cover
    from fido.github import RealGitHubFactory
    from fido.infra import RealClock, RealProcessRunner
    from fido.server import run

    run(github_factory=RealGitHubFactory(RealProcessRunner(), RealClock()))


if __name__ == "__main__":  # pragma: no cover
    main()
