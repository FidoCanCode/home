"""Regression tests for the handler_preemption product-state oracle.

The oracle models legacy in-memory handler demand, durable queued demand, and
provider interrupt request state as independent fields.  Worker admission is
controlled only by the legacy and durable demand fields; provider interrupt is
observable ordering state, not authority for whether demand exists.
"""

from fido.rocq.handler_preemption import (
    DurableDemandDrained,
    DurableDemandRecorded,
    DurableEmpty,
    DurableNonEmpty,
    HandlerDone,
    InterruptNotRequested,
    InterruptRequested,
    InterruptWasRequested,
    LegacyEmpty,
    LegacyNonEmpty,
    State,
    WebhookArrives,
    WorkerTurnStart,
    durable_state,
    empty_state,
    legacy_state,
    mixed_state,
    preempted_durable_state,
    transition,
)


def _state(
    *,
    legacy: type[LegacyEmpty] | type[LegacyNonEmpty],
    durable: type[DurableEmpty] | type[DurableNonEmpty],
    interrupt: type[InterruptNotRequested] | type[InterruptWasRequested],
) -> State:
    return State(
        legacy_demand=legacy(),
        durable_demand=durable(),
        provider_interrupt=interrupt(),
    )


def _assert_state(
    state: State | None,
    *,
    legacy: type[LegacyEmpty] | type[LegacyNonEmpty],
    durable: type[DurableEmpty] | type[DurableNonEmpty],
    interrupt: type[InterruptNotRequested] | type[InterruptWasRequested],
) -> State:
    assert state is not None
    assert isinstance(state.legacy_demand, legacy)
    assert isinstance(state.durable_demand, durable)
    assert isinstance(state.provider_interrupt, interrupt)
    return state


def test_worker_turn_rejected_from_legacy_demand() -> None:
    """WorkerTurnStart is rejected while legacy handler demand is non-empty."""
    assert transition(legacy_state, WorkerTurnStart()) is None


def test_worker_turn_rejected_from_durable_demand() -> None:
    """WorkerTurnStart is rejected while durable demand is queued."""
    assert transition(durable_state, WorkerTurnStart()) is None


def test_worker_turn_rejected_from_preempted_durable_demand() -> None:
    """WorkerTurnStart is rejected after durable demand requests interrupt."""
    assert transition(preempted_durable_state, WorkerTurnStart()) is None


def test_worker_turn_rejected_from_mixed_demand() -> None:
    """WorkerTurnStart is rejected when both demand sources are non-empty."""
    assert transition(mixed_state, WorkerTurnStart()) is None


def test_worker_turn_accepted_when_demands_empty_despite_interrupt() -> None:
    """Interrupt state does not block worker admission after demand drains."""
    interrupted_empty = _state(
        legacy=LegacyEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )

    result = transition(interrupted_empty, WorkerTurnStart())

    assert result == interrupted_empty


def test_interrupt_rejected_without_durable_demand() -> None:
    """InterruptRequested is rejected until demand has been durably recorded."""
    assert transition(empty_state, InterruptRequested()) is None
    assert transition(legacy_state, InterruptRequested()) is None


def test_durable_record_then_interrupt_sets_interrupt_field() -> None:
    """DurableDemandRecorded precedes InterruptRequested."""
    recorded = transition(empty_state, DurableDemandRecorded())
    recorded = _assert_state(
        recorded,
        legacy=LegacyEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptNotRequested,
    )

    _assert_state(
        transition(recorded, InterruptRequested()),
        legacy=LegacyEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptWasRequested,
    )


def test_handler_done_rejected_without_legacy_demand() -> None:
    """HandlerDone is rejected when no legacy handler demand exists."""
    assert transition(empty_state, HandlerDone()) is None
    assert transition(durable_state, HandlerDone()) is None


def test_handler_done_clears_only_legacy_demand() -> None:
    """HandlerDone remains valid while durable demand exists."""
    _assert_state(
        transition(mixed_state, HandlerDone()),
        legacy=LegacyEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptWasRequested,
    )


def test_durable_drain_clears_only_durable_demand() -> None:
    """DurableDemandDrained must not clear legacy handler demand."""
    _assert_state(
        transition(mixed_state, DurableDemandDrained()),
        legacy=LegacyNonEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )


def test_worker_turn_accepted_from_empty_state() -> None:
    """WorkerTurnStart from empty demand preserves the state."""
    assert transition(empty_state, WorkerTurnStart()) == empty_state


def test_repeated_worker_turns_from_empty_state() -> None:
    """Repeated WorkerTurnStart calls from empty demand preserve the state."""
    state = empty_state
    for _ in range(5):
        result = transition(state, WorkerTurnStart())
        assert result == state
        state = result


def test_webhook_arrival_records_legacy_without_losing_durable() -> None:
    """WebhookArrives remains valid while durable demand exists."""
    _assert_state(
        transition(durable_state, WebhookArrives()),
        legacy=LegacyNonEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptNotRequested,
    )


def test_durable_record_preserves_legacy_demand() -> None:
    """DurableDemandRecorded must not clear legacy demand."""
    _assert_state(
        transition(legacy_state, DurableDemandRecorded()),
        legacy=LegacyNonEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptNotRequested,
    )


def test_full_mixed_lifecycle_blocks_until_both_demands_drain() -> None:
    """Worker admission waits for legacy and durable demand to drain."""
    state = transition(empty_state, WebhookArrives())
    state = _assert_state(
        state,
        legacy=LegacyNonEmpty,
        durable=DurableEmpty,
        interrupt=InterruptNotRequested,
    )
    state = transition(state, DurableDemandRecorded())
    state = _assert_state(
        state,
        legacy=LegacyNonEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptNotRequested,
    )
    state = transition(state, InterruptRequested())
    state = _assert_state(
        state,
        legacy=LegacyNonEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptWasRequested,
    )

    assert transition(state, WorkerTurnStart()) is None
    handler_done = transition(state, HandlerDone())
    handler_done = _assert_state(
        handler_done,
        legacy=LegacyEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptWasRequested,
    )
    assert transition(handler_done, WorkerTurnStart()) is None
    drained = transition(handler_done, DurableDemandDrained())
    drained = _assert_state(
        drained,
        legacy=LegacyEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )
    assert transition(drained, WorkerTurnStart()) == drained


def test_mixed_handler_done_first_still_blocks_until_durable_drains() -> None:
    """Draining legacy demand first leaves durable demand authoritative."""
    state = transition(mixed_state, HandlerDone())
    state = _assert_state(
        state,
        legacy=LegacyEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptWasRequested,
    )

    assert transition(state, WorkerTurnStart()) is None
    state = transition(state, DurableDemandDrained())
    state = _assert_state(
        state,
        legacy=LegacyEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )
    assert transition(state, WorkerTurnStart()) == state


def test_mixed_durable_drain_first_still_blocks_until_handler_done() -> None:
    """Draining durable demand first leaves legacy demand authoritative."""
    state = transition(mixed_state, DurableDemandDrained())
    state = _assert_state(
        state,
        legacy=LegacyNonEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )

    assert transition(state, WorkerTurnStart()) is None
    state = transition(state, HandlerDone())
    state = _assert_state(
        state,
        legacy=LegacyEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )
    assert transition(state, WorkerTurnStart()) == state


def test_durable_first_mixed_path_blocks_until_both_demands_drain() -> None:
    """Durable demand can predate legacy demand without losing either blocker."""
    state = transition(empty_state, DurableDemandRecorded())
    state = _assert_state(
        state,
        legacy=LegacyEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptNotRequested,
    )
    state = transition(state, InterruptRequested())
    state = _assert_state(
        state,
        legacy=LegacyEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptWasRequested,
    )
    state = transition(state, WebhookArrives())
    state = _assert_state(
        state,
        legacy=LegacyNonEmpty,
        durable=DurableNonEmpty,
        interrupt=InterruptWasRequested,
    )

    assert transition(state, WorkerTurnStart()) is None
    state = transition(state, DurableDemandDrained())
    state = _assert_state(
        state,
        legacy=LegacyNonEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )
    assert transition(state, WorkerTurnStart()) is None
    state = transition(state, HandlerDone())
    state = _assert_state(
        state,
        legacy=LegacyEmpty,
        durable=DurableEmpty,
        interrupt=InterruptWasRequested,
    )
    assert transition(state, WorkerTurnStart()) == state
