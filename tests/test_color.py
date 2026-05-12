"""Tests for fido.color — ANSI color helper."""

import pytest

from fido.color import (
    _CODES,
    _RESET,
    BOLD,
    CYAN,
    DARK_GRAY,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    RED_BOLD,
    YELLOW,
    ColorRenderer,
    color,
    color_enabled,
)

# ---------------------------------------------------------------------------
# color_enabled
# ---------------------------------------------------------------------------


class TestColorEnabled:
    def test_force_color_enables(self) -> None:
        r = ColorRenderer(environ={"FORCE_COLOR": "1"}, is_tty=False)
        assert r.color_enabled() is True

    def test_force_color_overrides_no_color(self) -> None:
        r = ColorRenderer(environ={"FORCE_COLOR": "1", "NO_COLOR": ""}, is_tty=False)
        assert r.color_enabled() is True

    def test_no_color_disables(self) -> None:
        r = ColorRenderer(environ={"NO_COLOR": ""}, is_tty=False)
        assert r.color_enabled() is False

    def test_no_color_any_value_disables(self) -> None:
        r = ColorRenderer(environ={"NO_COLOR": "1"}, is_tty=False)
        assert r.color_enabled() is False

    def test_tty_enables(self) -> None:
        r = ColorRenderer(environ={}, is_tty=True)
        assert r.color_enabled() is True

    def test_non_tty_disables(self) -> None:
        r = ColorRenderer(environ={}, is_tty=False)
        assert r.color_enabled() is False

    def test_force_color_wrong_value_falls_through_to_tty(self) -> None:
        r = ColorRenderer(environ={"FORCE_COLOR": "0"}, is_tty=False)
        assert r.color_enabled() is False

    def test_module_level_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert color_enabled() is True


# ---------------------------------------------------------------------------
# color()
# ---------------------------------------------------------------------------


class TestColor:
    def _enabled(self) -> ColorRenderer:
        """Renderer with color forced on."""
        return ColorRenderer(environ={"FORCE_COLOR": "1"}, is_tty=False)

    def _disabled(self) -> ColorRenderer:
        """Renderer with color forced off."""
        return ColorRenderer(environ={"NO_COLOR": ""}, is_tty=False)

    def test_disabled_returns_text_unchanged(self) -> None:
        assert self._disabled().color(BOLD, "hello") == "hello"

    def test_unknown_style_returns_text_unchanged(self) -> None:
        assert self._enabled().color("neon_rainbow", "oops") == "oops"

    @pytest.mark.parametrize(
        "style",
        [BOLD, DIM, RED, RED_BOLD, CYAN, MAGENTA, GREEN, YELLOW, DARK_GRAY],
    )
    def test_each_style_wraps_with_ansi(self, style: str) -> None:
        result = self._enabled().color(style, "text")
        assert result == f"{_CODES[style]}text{_RESET}"

    def testcolor_enabled_bold(self) -> None:
        assert self._enabled().color(BOLD, "hi") == "\033[1mhi\033[0m"

    def testcolor_enabled_dim(self) -> None:
        assert self._enabled().color(DIM, "quiet") == "\033[2mquiet\033[0m"

    def testcolor_enabled_red(self) -> None:
        assert self._enabled().color(RED, "danger") == "\033[31mdanger\033[0m"

    def testcolor_enabled_red_bold(self) -> None:
        assert self._enabled().color(RED_BOLD, "alarm") == "\033[1;31malarm\033[0m"

    def testcolor_enabled_cyan(self) -> None:
        assert self._enabled().color(CYAN, "#42") == "\033[36m#42\033[0m"

    def testcolor_enabled_magenta(self) -> None:
        assert self._enabled().color(MAGENTA, "#99") == "\033[35m#99\033[0m"

    def testcolor_enabled_green(self) -> None:
        assert self._enabled().color(GREEN, "worker") == "\033[32mworker\033[0m"

    def testcolor_enabled_yellow(self) -> None:
        assert self._enabled().color(YELLOW, "webhook") == "\033[33mwebhook\033[0m"

    def testcolor_enabled_dark_gray(self) -> None:
        assert self._enabled().color(DARK_GRAY, "paused") == "\033[90mpaused\033[0m"

    def test_module_level_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert color(BOLD, "hi") == "\033[1mhi\033[0m"


# ---------------------------------------------------------------------------
# wrap_raw / rgb_fg / rgb_bg / wrap_bg_line  (provider-color feature)
# ---------------------------------------------------------------------------


class TestRawWrapping:
    def _enabled(self) -> ColorRenderer:
        return ColorRenderer(environ={"FORCE_COLOR": "1"}, is_tty=False)

    def _disabled(self) -> ColorRenderer:
        return ColorRenderer(environ={"NO_COLOR": ""}, is_tty=False)

    def test_rgb_fg_emits_truecolor_escape(self) -> None:
        from fido.color import rgb_fg

        assert rgb_fg(255, 160, 60) == "\033[38;2;255;160;60m"

    def test_rgb_bg_emits_truecolor_escape(self) -> None:
        from fido.color import rgb_bg

        assert rgb_bg(30, 15, 0) == "\033[48;2;30;15;0m"

    def test_wrap_raw_wraps_when_enabled(self) -> None:
        from fido.color import rgb_fg

        result = self._enabled().wrap_raw(rgb_fg(10, 20, 30), "x")
        assert result == f"\033[38;2;10;20;30mx{_RESET}"

    def test_wrap_raw_returns_text_when_disabled(self) -> None:
        from fido.color import rgb_fg

        assert self._disabled().wrap_raw(rgb_fg(10, 20, 30), "x") == "x"

    def test_wrap_raw_ignores_empty_escape(self) -> None:
        assert self._enabled().wrap_raw("", "x") == "x"

    def test_wrap_bg_line_applies_bg_across_inner_resets(self) -> None:
        """Inner `_RESET`s must not punch holes in the background."""
        from fido.color import rgb_bg

        bg = rgb_bg(30, 15, 0)
        # A pre-styled line: bold "A" then plain "B".
        line = f"\033[1mA{_RESET}B"
        result = self._enabled().wrap_bg_line(bg, line)
        # After every inner reset, bg is re-applied; one final reset closes.
        assert result == f"{bg}\033[1mA{_RESET}{bg}B{_RESET}"

    def test_wrap_bg_line_no_inner_reset(self) -> None:
        from fido.color import rgb_bg

        bg = rgb_bg(30, 15, 0)
        assert self._enabled().wrap_bg_line(bg, "plain") == f"{bg}plain{_RESET}"

    def test_wrap_bg_line_disabled_returns_line_unchanged(self) -> None:
        from fido.color import rgb_bg

        assert self._disabled().wrap_bg_line(rgb_bg(30, 15, 0), "plain") == "plain"

    def test_wrap_bg_line_empty_escape_returns_line_unchanged(self) -> None:
        assert self._enabled().wrap_bg_line("", "plain") == "plain"

    def test_module_level_wrap_raw_delegates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fido.color import rgb_fg, wrap_raw

        monkeypatch.setenv("FORCE_COLOR", "1")
        assert wrap_raw(rgb_fg(10, 20, 30), "x") == f"\033[38;2;10;20;30mx{_RESET}"

    def test_module_level_wrap_bg_line_delegates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fido.color import rgb_bg, wrap_bg_line

        monkeypatch.setenv("FORCE_COLOR", "1")
        bg = rgb_bg(30, 15, 0)
        assert wrap_bg_line(bg, "plain") == f"{bg}plain{_RESET}"
