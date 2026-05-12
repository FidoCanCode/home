"""ANSI color helper for terminal output.

Color is enabled when:
- ``FORCE_COLOR=1`` is set in the environment, OR
- stdout is a TTY and ``NO_COLOR`` is not set.

``NO_COLOR`` follows https://no-color.org — presence of the variable (any
value) disables color.  ``FORCE_COLOR=1`` overrides both TTY detection and
``NO_COLOR`` so that tools like ``watch -c`` work without extra flags.
"""

import os
import sys
from collections.abc import Mapping

# Semantic style names
BOLD = "bold"
DIM = "dim"
RED = "red"
RED_BOLD = "red_bold"
CYAN = "cyan"
MAGENTA = "magenta"
GREEN = "green"
YELLOW = "yellow"
DARK_GRAY = "dark_gray"
GREEN_BG = "green_bg"
YELLOW_BG = "yellow_bg"

# ANSI escape sequences
_RESET = "\033[0m"
_CODES: dict[str, str] = {
    BOLD: "\033[1m",
    DIM: "\033[2m",
    RED: "\033[31m",
    RED_BOLD: "\033[1;31m",
    CYAN: "\033[36m",
    MAGENTA: "\033[35m",
    GREEN: "\033[32m",
    YELLOW: "\033[33m",
    DARK_GRAY: "\033[90m",
    GREEN_BG: "\033[30;42m",
    YELLOW_BG: "\033[30;43m",
}


def rgb_fg(r: int, g: int, b: int) -> str:
    """Return a truecolor-foreground ANSI escape for (r, g, b)."""
    return f"\033[38;2;{r};{g};{b}m"


def rgb_bg(r: int, g: int, b: int) -> str:
    """Return a truecolor-background ANSI escape for (r, g, b)."""
    return f"\033[48;2;{r};{g};{b}m"


class ColorRenderer:
    """Renders ANSI color sequences with injectable environment and TTY state.

    Parameters
    ----------
    environ:
        Environment mapping to consult for ``FORCE_COLOR`` / ``NO_COLOR``.
        Defaults to ``os.environ`` (live, not a snapshot).
    is_tty:
        Override for ``sys.stdout.isatty()``.  When *None* (the default),
        the live call is made each time :meth:`color_enabled` is invoked.
        Pass ``True`` or ``False`` in tests to avoid touching the real stdout.
    """

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        is_tty: bool | None = None,
    ) -> None:
        self._environ: Mapping[str, str] = (
            environ if environ is not None else os.environ
        )
        self._is_tty_override = is_tty

    def color_enabled(self) -> bool:
        """Return True if ANSI color output should be used."""
        if self._environ.get("FORCE_COLOR") == "1":
            return True
        if "NO_COLOR" in self._environ:
            return False
        if self._is_tty_override is not None:
            return self._is_tty_override
        return sys.stdout.isatty()

    def color(self, style: str, text: str) -> str:
        """Wrap *text* in ANSI escape codes for *style* if color is enabled.

        Returns *text* unchanged when color is disabled or *style* is unknown.
        """
        if not self.color_enabled():
            return text
        code = _CODES.get(style, "")
        if not code:
            return text
        return f"{code}{text}{_RESET}"

    def wrap_raw(self, escape: str, text: str) -> str:
        """Wrap *text* in a raw ANSI *escape* sequence when color is enabled.

        Lower-level than :meth:`color`: accepts any pre-built ANSI escape
        (e.g. from :func:`rgb_fg`/:func:`rgb_bg`) so callers can render
        provider-specific truecolor without registering a named style for
        every provider.  Returns *text* unchanged when color is disabled or
        *escape* is empty.
        """
        if not self.color_enabled():
            return text
        if not escape:
            return text
        return f"{escape}{text}{_RESET}"

    def wrap_bg_line(self, bg_escape: str, line: str) -> str:
        """Apply *bg_escape* across a pre-styled *line* so inner resets don't
        punch holes in the background.

        Inner ``\\x1b[0m`` resets clear every attribute, including the
        background — so a naive ``f"{bg}{line}{_RESET}"`` would lose the tint
        after the first styled token.  This helper re-applies ``bg_escape``
        after every internal reset so the bg persists across the entire line,
        then closes with a single final reset.

        Returns *line* unchanged when color is disabled or *bg_escape* is
        empty.
        """
        if not self.color_enabled() or not bg_escape:
            return line
        if _RESET in line:
            line = line.replace(_RESET, f"{_RESET}{bg_escape}")
        return f"{bg_escape}{line}{_RESET}"


_renderer = ColorRenderer()


def color_enabled() -> bool:
    """Return True if ANSI color output should be used."""
    return _renderer.color_enabled()


def color(style: str, text: str) -> str:
    """Wrap *text* in ANSI escape codes for *style* if color is enabled.

    Returns *text* unchanged when color is disabled or *style* is unknown.
    """
    return _renderer.color(style, text)


def wrap_raw(escape: str, text: str) -> str:
    """Wrap *text* in a raw ANSI *escape* sequence when color is enabled.

    Lower-level than :func:`color`: accepts any pre-built ANSI escape
    (e.g. from :func:`rgb_fg`/:func:`rgb_bg`) so callers can render
    provider-specific truecolor without registering a named style for
    every provider.  Returns *text* unchanged when color is disabled or
    *escape* is empty.
    """
    return _renderer.wrap_raw(escape, text)


def wrap_bg_line(bg_escape: str, line: str) -> str:
    """Apply *bg_escape* across a pre-styled *line* so inner resets don't
    punch holes in the background.

    Inner ``\\x1b[0m`` resets clear every attribute, including the
    background — so a naive ``f"{bg}{line}{_RESET}"`` would lose the tint
    after the first styled token.  This helper re-applies ``bg_escape``
    after every internal reset so the bg persists across the entire line,
    then closes with a single final reset.

    Returns *line* unchanged when color is disabled or *bg_escape* is
    empty.
    """
    return _renderer.wrap_bg_line(bg_escape, line)
