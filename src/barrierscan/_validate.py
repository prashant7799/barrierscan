"""Input checks shared by `first_touch` and `brute_force_one`.

These functions only check and convert inputs. They hold no labelling
rules, so the fast kernel and the slow oracle stay independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENTRY_MODES = ("next_open", "close")
TIE_RULES = ("sl_first", "tp_first")


@dataclass(frozen=True, eq=False)
class CheckedInputs:
    """Inputs after checking. Every per-bar array has one value per bar."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tp: np.ndarray
    sl: np.ndarray
    offset: np.ndarray
    entry_mode: str  # "next_open", "close" or "custom"
    entry_prices: np.ndarray | None  # only set when entry_mode == "custom"
    horizon: int  # capped at the number of bars
    tie: str
    gap_exit: bool
    events: np.ndarray  # int64 bar positions, strictly increasing


def _reject_complex(values, name: str) -> None:
    # numpy would silently drop the imaginary part when converting to float.
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must hold real numbers, not complex numbers")


def _price_array(values, name: str) -> np.ndarray:
    _reject_complex(values, name)
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got {array.ndim} dimensions")
    bad = np.flatnonzero(~np.isfinite(array))
    if bad.size > 0:
        raise ValueError(f"{name} must be finite; first bad bar is {bad[0]}")
    return array


def _per_bar(values, name: str, n_bars: int) -> np.ndarray:
    """Turn a scalar or a length-n array into a float64 array of length n."""
    _reject_complex(values, name)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return np.full(n_bars, float(array))
    if array.ndim == 1 and array.shape[0] == n_bars:
        return array
    raise ValueError(
        f"{name} must be a scalar or a 1-D array with one value per bar ({n_bars}), "
        f"got shape {array.shape}"
    )


def _check_ohlc(open_, high, low, close) -> None:
    n_bars = open_.shape[0]
    for name, array in (("high", high), ("low", low), ("close", close)):
        if array.shape[0] != n_bars:
            raise ValueError(
                f"open, high, low and close must have the same length; "
                f"open has {n_bars}, {name} has {array.shape[0]}"
            )
    if n_bars < 2:
        raise ValueError("need at least 2 bars: an event bar and one bar after it")

    too_low_high = np.flatnonzero(high < np.maximum(open_, close))
    if too_low_high.size > 0:
        raise ValueError(
            f"high must be >= max(open, close) on every bar; first bad bar is {too_low_high[0]}"
        )
    too_high_low = np.flatnonzero(low > np.minimum(open_, close))
    if too_high_low.size > 0:
        raise ValueError(
            f"low must be <= min(open, close) on every bar; first bad bar is {too_high_low[0]}"
        )


def _check_events(events, n_bars: int) -> np.ndarray:
    if events is None:
        # Every bar that has at least one bar after it.
        return np.arange(n_bars - 1, dtype=np.int64)

    array = np.asarray(events)
    if array.ndim != 1:
        raise ValueError("events must be a 1-D array of bar positions")
    if array.size == 0:
        return np.zeros(0, dtype=np.int64)
    if array.dtype == np.bool_:
        raise TypeError(
            "events must hold integer bar positions, not a True/False mask; "
            "use np.flatnonzero(mask) to turn a mask into positions"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"events must hold integer bar positions, got dtype {array.dtype}")

    array = array.astype(np.int64)
    if array.min() < 0 or array.max() > n_bars - 2:
        raise ValueError(
            f"events must be between 0 and {n_bars - 2} (the last bar has no bar after it)"
        )
    if np.any(np.diff(array) <= 0):
        raise ValueError("events must be strictly increasing (sorted, no duplicates)")
    return array


def _check_positive_at_events(values: np.ndarray, name: str, events: np.ndarray) -> None:
    at_events = values[events]
    bad = np.flatnonzero(~(np.isfinite(at_events) & (at_events > 0)))
    if bad.size > 0:
        raise ValueError(
            f"{name} must be finite and > 0 at every event; bad at bar {events[bad[0]]}"
        )


def check_inputs(
    open_,
    high,
    low,
    close,
    *,
    tp,
    sl,
    horizon,
    entry,
    tie,
    entry_offset,
    gap_exit,
    events,
) -> CheckedInputs:
    """Check every input and return clean float64/int64 arrays."""
    open_ = _price_array(open_, "open")
    high = _price_array(high, "high")
    low = _price_array(low, "low")
    close = _price_array(close, "close")
    _check_ohlc(open_, high, low, close)
    n_bars = open_.shape[0]

    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError(f"horizon must be an int, got {type(horizon).__name__}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if tie not in TIE_RULES:
        raise ValueError(f"tie must be one of {TIE_RULES}, got {tie!r}")

    if not isinstance(gap_exit, (bool, np.bool_)):
        raise TypeError("gap_exit must be True or False")

    event_bars = _check_events(events, n_bars)

    tp_per_bar = _per_bar(tp, "tp", n_bars)
    sl_per_bar = _per_bar(sl, "sl", n_bars)
    _check_positive_at_events(tp_per_bar, "tp", event_bars)
    _check_positive_at_events(sl_per_bar, "sl", event_bars)

    offset_per_bar = _per_bar(entry_offset, "entry_offset", n_bars)
    offset_at_events = offset_per_bar[event_bars]
    bad = np.flatnonzero(~(np.isfinite(offset_at_events) & (offset_at_events >= 0)))
    if bad.size > 0:
        raise ValueError(
            f"entry_offset must be finite and >= 0 at every event; bad at bar {event_bars[bad[0]]}"
        )

    entry_prices = None
    if isinstance(entry, str):
        if entry not in ENTRY_MODES:
            raise ValueError(
                f"entry must be 'next_open', 'close' or an array of entry prices, got {entry!r}"
            )
        entry_mode = entry
    else:
        if np.ndim(entry) == 0:
            raise ValueError("a custom entry must be an array with one price per bar")
        entry_mode = "custom"
        entry_prices = _per_bar(entry, "entry", n_bars)
        bad = np.flatnonzero(~np.isfinite(entry_prices[event_bars]))
        if bad.size > 0:
            raise ValueError(
                f"entry prices must be finite at every event; bad at bar {event_bars[bad[0]]}"
            )

    return CheckedInputs(
        open=open_,
        high=high,
        low=low,
        close=close,
        tp=tp_per_bar,
        sl=sl_per_bar,
        offset=offset_per_bar,
        entry_mode=entry_mode,
        entry_prices=entry_prices,
        # A horizon longer than the data just scans to the last bar, so we cap
        # it at n_bars. Without the cap, i + horizon could overflow the 64-bit
        # integers inside the numba kernel and give wrong results.
        horizon=min(int(horizon), n_bars),
        tie=tie,
        gap_exit=bool(gap_exit),
        events=event_bars,
    )
