"""Small helpers for choosing barriers and reading results.

- `average_true_range`: a trailing measure of how much bars move.
- `vol_barriers`: tp/sl arrays that scale with volatility, with a floor.
- `entry_offset`: half the spread plus slippage, for `first_touch(entry_offset=...)`.
- `breakeven_hit_rate`: the target hit rate at which a target/stop trade breaks even.
"""

from __future__ import annotations

import numpy as np


def _as_1d_float(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def average_true_range(high, low, close, window: int = 14) -> np.ndarray:
    """Trailing average true range (ATR), one value per bar.

    True range of bar i is the largest of: high - low, |high - previous close|
    and |low - previous close|. The first bar has no previous close, so its
    true range is high - low.

    ATR at bar i is the plain mean of the true ranges of bars
    max(0, i - window + 1) .. i. It uses bar i and earlier bars only, so it
    is safe to use for an event at bar i. The first window - 1 bars average
    the bars that exist so far.
    """
    high = _as_1d_float(high, "high")
    low = _as_1d_float(low, "low")
    close = _as_1d_float(close, "close")
    if not (high.shape == low.shape == close.shape):
        raise ValueError("high, low and close must have the same length")
    if high.shape[0] == 0:
        raise ValueError("need at least one bar")
    if np.any(high < low):
        raise ValueError("high must be >= low on every bar")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be an int >= 1")

    n_bars = high.shape[0]
    previous_close = np.empty(n_bars)
    previous_close[0] = close[0]  # makes the first true range equal high - low
    previous_close[1:] = close[:-1]
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)),
    )

    # Trailing mean from a running total: the sum of true ranges over bars
    # a+1 .. b is running_total[b] - running_total[a].
    running_total = np.cumsum(true_range)
    atr = np.empty(n_bars)
    warm_up = min(window, n_bars)
    atr[:warm_up] = running_total[:warm_up] / np.arange(1, warm_up + 1)
    if n_bars > window:
        atr[window:] = (running_total[window:] - running_total[:-window]) / window
    return atr


def vol_barriers(
    high,
    low,
    close,
    *,
    window: int = 14,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    floor=0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Volatility-scaled tp and sl distances, with a minimum stop size.

    sl = max(sl_mult * ATR, floor) and tp = sl * tp_mult / sl_mult, so the
    target-to-stop ratio stays tp_mult / sl_mult on every bar.

    ``floor`` is a scalar or one value per bar, for example three times the
    typical spread. It stops barriers from shrinking to almost nothing in
    quiet periods. Use floor > 0 if your data can have flat stretches,
    because `first_touch` needs tp and sl > 0.

    Returns ``(tp, sl)``, two float64 arrays with one value per bar.
    """
    if not (np.isfinite(tp_mult) and tp_mult > 0 and np.isfinite(sl_mult) and sl_mult > 0):
        raise ValueError("tp_mult and sl_mult must be finite and > 0")
    atr = average_true_range(high, low, close, window)

    floor = np.asarray(floor, dtype=np.float64)
    if floor.ndim not in (0, 1) or (floor.ndim == 1 and floor.shape != atr.shape):
        raise ValueError("floor must be a scalar or one value per bar")
    if not np.all(np.isfinite(floor)) or np.any(floor < 0):
        raise ValueError("floor must be finite and >= 0")

    sl = np.maximum(sl_mult * atr, floor)
    tp = sl * (tp_mult / sl_mult)
    return tp, sl


def entry_offset(spread, slippage=0.0):
    """Half the bid-ask spread plus slippage.

    Pass the result to ``first_touch(entry_offset=...)``. A long then fills
    this far above the reference price and a short this far below it.
    ``spread`` and ``slippage`` are scalars or arrays in price units, >= 0.
    Returns a float for scalar inputs, otherwise a float64 array.
    """
    spread = np.asarray(spread, dtype=np.float64)
    slippage = np.asarray(slippage, dtype=np.float64)
    for name, values in (("spread", spread), ("slippage", slippage)):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name} must be finite and >= 0")
    offset = spread / 2.0 + slippage
    if offset.ndim == 0:
        return float(offset)
    return offset


def breakeven_hit_rate(tp, sl, cost=0.0):
    """Target hit rate at which a trade with two outcomes breaks even.

    Each trade either gains tp or loses sl, and pays ``cost`` (same units).
    Solving p * tp - (1 - p) * sl - cost = 0 gives

        p = (sl + cost) / (tp + sl)

    Example: tp = 3, sl = 1, cost = 0 gives 0.25. A hit rate above this is
    needed to make money before any other costs. It ignores timeouts.
    Returns a float for scalar inputs, otherwise a float64 array.
    """
    tp = np.asarray(tp, dtype=np.float64)
    sl = np.asarray(sl, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if not (np.all(np.isfinite(tp)) and np.all(tp > 0)):
        raise ValueError("tp must be finite and > 0")
    if not (np.all(np.isfinite(sl)) and np.all(sl > 0)):
        raise ValueError("sl must be finite and > 0")
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost must be finite")
    rate = (sl + cost) / (tp + sl)
    if rate.ndim == 0:
        return float(rate)
    return rate
