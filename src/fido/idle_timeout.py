"""Shared idle-timeout bookkeeping for streaming provider turns."""

from fido.infra import Clock, RealClock


class IdleDeadline:
    """Track provider turn idleness using bounded poll slices.

    Provider loops should call :meth:`poll_timeout` before each blocking read and
    :meth:`reset` whenever real provider activity arrives.  This keeps
    preemption responsive while allowing long model turns that emit occasional
    activity.
    """

    def __init__(
        self,
        timeout: float,
        *,
        poll_interval: float,
        clock: Clock | None = None,
    ) -> None:
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._clock = clock if clock is not None else RealClock()
        self._last_activity = self._clock.monotonic()

    def reset(self) -> None:
        self._last_activity = self._clock.monotonic()

    def remaining(self) -> float:
        return self._timeout - (self._clock.monotonic() - self._last_activity)

    def poll_timeout(self) -> float:
        return min(self.remaining(), self._poll_interval)

    def poll_timeout_or_expired(self) -> float | None:
        remaining = self.remaining()
        if remaining <= 0:
            return None
        return min(remaining, self._poll_interval)

    def expired(self) -> bool:
        return self.remaining() <= 0
