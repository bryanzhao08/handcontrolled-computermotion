"""
macOS system action wrappers.

All public functions are registered in ACTION_MAP so they can be dispatched
by name from config.yaml without any code changes.
"""
from __future__ import annotations

import inspect
import logging
import subprocess

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _osascript(script: str) -> bool:
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        log.warning("osascript failed: %s", e.stderr.decode().strip())
        return False


# ── Volume ────────────────────────────────────────────────────────────────────

def _get_volume() -> int:
    r = subprocess.run(
        ["osascript", "-e", "output volume of (get volume settings)"],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 50


def volume_up(delta: int = 5) -> None:
    new = min(100, _get_volume() + delta)
    _osascript(f"set volume output volume {new}")
    log.info("Volume → %d", new)


def volume_down(delta: int = 5) -> None:
    new = max(0, _get_volume() - delta)
    _osascript(f"set volume output volume {new}")
    log.info("Volume → %d", new)


def mute_toggle() -> None:
    _osascript("set volume output muted not (output muted of (get volume settings))")
    log.info("Mute toggled")


def volume_min() -> None:
    """Set volume all the way to 0."""
    _osascript("set volume output volume 0")
    log.info("Volume → 0")


def volume_max() -> None:
    """Set volume all the way to 100."""
    _osascript("set volume output volume 100")
    log.info("Volume → 100")


# ── Brightness ────────────────────────────────────────────────────────────────
# Uses synthetic key events (key code 144 = brightness up, 145 = brightness down).
# Each press moves brightness ~6.25%, so delta=10 ≈ 2 key presses.

def _brightness_keys(delta_pct: float) -> None:
    key_code = 144 if delta_pct > 0 else 145
    steps = max(1, round(abs(delta_pct) / 6.25))
    for _ in range(steps):
        subprocess.run(
            ["osascript", "-e", f"tell application \"System Events\" to key code {key_code}"],
            capture_output=True,
        )


def brightness_up(delta: int = 10) -> None:
    _brightness_keys(+delta)
    log.info("Brightness up ~%d%%", delta)


def brightness_down(delta: int = 10) -> None:
    _brightness_keys(-delta)
    log.info("Brightness down ~%d%%", delta)


# ── Misc ──────────────────────────────────────────────────────────────────────

def mission_control() -> None:
    _osascript('tell application "System Events" to key code 160')


def screenshot() -> None:
    subprocess.run(["screencapture", "-ix", "/tmp/gesture_screenshot.png"])


def no_action() -> None:
    pass


# ── Dispatcher ────────────────────────────────────────────────────────────────

ACTION_MAP: dict[str, callable] = {
    "volume_up":       volume_up,
    "volume_down":     volume_down,
    "volume_min":      volume_min,
    "volume_max":      volume_max,
    "mute_toggle":     mute_toggle,
    "brightness_up":   brightness_up,
    "brightness_down": brightness_down,
    "mission_control": mission_control,
    "screenshot":      screenshot,
    "none":            no_action,
}


def dispatch(action_name: str, delta: int = 5) -> None:
    """Call the named action, passing delta if the function accepts it."""
    fn = ACTION_MAP.get(action_name)
    if fn is None:
        log.warning("Unknown action: '%s'", action_name)
        return
    try:
        sig = inspect.signature(fn)
        if "delta" in sig.parameters:
            fn(delta=delta)
        else:
            fn()
    except Exception as e:
        log.error("Action '%s' raised: %s", action_name, e)
