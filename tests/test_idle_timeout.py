from fido.idle_timeout import IdleDeadline


class _FakeClock:
    """Controllable :class:`~fido.infra.Clock` for idle-timeout tests.

    ``now_value`` is mutable so tests can advance time after construction.
    Only :meth:`monotonic` is exercised by :class:`IdleDeadline`; the
    other methods are intentionally absent.
    """

    def __init__(self, now_value: float = 0.0) -> None:
        self.now_value = now_value
        self.call_count = 0

    def monotonic(self) -> float:
        self.call_count += 1
        return self.now_value


def test_poll_timeout_is_capped_by_poll_interval() -> None:
    clock = _FakeClock(0.0)
    deadline = IdleDeadline(30.0, poll_interval=1.0, clock=clock)

    assert deadline.poll_timeout() == 1.0


def test_poll_timeout_shrinks_to_remaining_idle_budget() -> None:
    clock = _FakeClock(9.75)
    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    clock.now_value = 19.5

    assert deadline.poll_timeout() == 0.25
    assert not deadline.expired()


def test_reset_extends_deadline_from_current_activity() -> None:
    clock = _FakeClock(0.0)
    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    clock.now_value = 9.0
    deadline.reset()
    clock.now_value = 18.0

    assert deadline.poll_timeout() == 1.0
    assert not deadline.expired()


def test_expired_after_full_idle_budget() -> None:
    clock = _FakeClock(0.0)
    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    clock.now_value = 10.0

    assert deadline.expired()


def test_poll_timeout_or_expired_samples_clock_once() -> None:
    clock = _FakeClock(0.0)
    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    clock.call_count = 0  # reset after __init__ call
    clock.now_value = 11.0

    assert deadline.poll_timeout_or_expired() is None
    assert clock.call_count == 1
