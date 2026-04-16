from __future__ import annotations

from datetime import UTC, datetime

from kennel.provider import (
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
    ProviderModel,
    ProviderPressureStatus,
)


class TestProviderLimitWindow:
    def test_pressure_returns_ratio(self) -> None:
        window = ProviderLimitWindow(name="requests", used=9, limit=10)
        assert window.pressure == 0.9

    def test_pressure_returns_none_when_used_missing(self) -> None:
        window = ProviderLimitWindow(name="requests", used=None, limit=10)
        assert window.pressure is None

    def test_pressure_returns_none_when_limit_missing(self) -> None:
        window = ProviderLimitWindow(name="requests", used=9, limit=None)
        assert window.pressure is None

    def test_pressure_returns_none_when_limit_not_positive(self) -> None:
        window = ProviderLimitWindow(name="requests", used=9, limit=0)
        assert window.pressure is None


class TestProviderLimitSnapshot:
    def test_closest_to_exhaustion_picks_highest_pressure(self) -> None:
        low = ProviderLimitWindow(name="tokens", used=20, limit=100)
        high = ProviderLimitWindow(name="requests", used=95, limit=100)
        snapshot = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE, windows=(low, high)
        )
        assert snapshot.closest_to_exhaustion() is high

    def test_closest_to_exhaustion_falls_back_to_first_window(self) -> None:
        first = ProviderLimitWindow(
            name="requests",
            resets_at=datetime(2026, 4, 16, tzinfo=UTC),
        )
        second = ProviderLimitWindow(name="tokens")
        snapshot = ProviderLimitSnapshot(
            provider=ProviderID.COPILOT_CLI,
            windows=(first, second),
        )
        assert snapshot.closest_to_exhaustion() is first

    def test_closest_to_exhaustion_returns_none_for_empty_snapshot(self) -> None:
        snapshot = ProviderLimitSnapshot(provider=ProviderID.GEMINI)
        assert snapshot.closest_to_exhaustion() is None


class TestProviderPressureStatus:
    def test_from_snapshot_uses_closest_window(self) -> None:
        low = ProviderLimitWindow(name="tokens", used=20, limit=100)
        high = ProviderLimitWindow(
            name="requests",
            used=96,
            limit=100,
            resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=UTC),
        )
        status = ProviderPressureStatus.from_snapshot(
            ProviderLimitSnapshot(
                provider=ProviderID.CLAUDE_CODE,
                windows=(low, high),
            )
        )
        assert status.provider is ProviderID.CLAUDE_CODE
        assert status.window_name == "requests"
        assert status.pressure == 0.96
        assert status.resets_at == datetime(2026, 4, 16, 7, 0, tzinfo=UTC)

    def test_level_is_warning_at_ninety_percent(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.9,
        )
        assert status.level == "warning"
        assert status.warning is True
        assert status.paused is False

    def test_level_is_paused_at_ninety_five_percent(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.95,
        )
        assert status.level == "paused"
        assert status.warning is False
        assert status.paused is True

    def test_level_is_unavailable_when_reason_present(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.COPILOT_CLI,
            pressure=0.99,
            unavailable_reason="limits unavailable",
        )
        assert status.level == "unavailable"

    def test_percent_used_rounds_to_nearest_whole_percent(self) -> None:
        status = ProviderPressureStatus(provider=ProviderID.CLAUDE_CODE, pressure=0.946)
        assert status.percent_used == 95

    def test_percent_used_is_none_when_pressure_unknown(self) -> None:
        status = ProviderPressureStatus(provider=ProviderID.CLAUDE_CODE)
        assert status.percent_used is None

    def test_level_is_ok_below_warning_threshold(self) -> None:
        status = ProviderPressureStatus(
            provider=ProviderID.CLAUDE_CODE,
            pressure=0.42,
        )
        assert status.level == "ok"


class TestProviderModel:
    def test_formats_and_compares_to_string(self) -> None:
        model = ProviderModel("gpt-5.4", "high")
        assert str(model) == "gpt-5.4"
        assert model == "gpt-5.4"

    def test_hash_and_model_equality_include_effort(self) -> None:
        model = ProviderModel("gpt-5.4", "high")
        same = ProviderModel("gpt-5.4", "high")
        different = ProviderModel("gpt-5.4", "medium")
        assert model == same
        assert hash(model) == hash(same)
        assert model != different

    def test_comparison_to_unrelated_type_is_false(self) -> None:
        assert ProviderModel("gpt-5.4") != object()
