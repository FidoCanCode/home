from fido.idle_timeout import IdleDeadline


def test_poll_timeout_is_capped_by_poll_interval() -> None:
    now = 0.0

    def clock() -> float:
        return now

    deadline = IdleDeadline(30.0, poll_interval=1.0, clock=clock)

    assert deadline.poll_timeout() == 1.0


def test_poll_timeout_shrinks_to_remaining_idle_budget() -> None:
    now = 9.75

    def clock() -> float:
        return now

    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    now = 19.5

    assert deadline.poll_timeout() == 0.25
    assert not deadline.expired()


def test_reset_extends_deadline_from_current_activity() -> None:
    now = 0.0

    def clock() -> float:
        return now

    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    now = 9.0
    deadline.reset()
    now = 18.0

    assert deadline.poll_timeout() == 1.0
    assert not deadline.expired()


def test_expired_after_full_idle_budget() -> None:
    now = 0.0

    def clock() -> float:
        return now

    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    now = 10.0

    assert deadline.expired()


def test_poll_timeout_or_expired_samples_clock_once() -> None:
    calls = 0
    now = 0.0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return now

    deadline = IdleDeadline(10.0, poll_interval=1.0, clock=clock)
    now = 11.0

    assert deadline.poll_timeout_or_expired() is None
    assert calls == 2
