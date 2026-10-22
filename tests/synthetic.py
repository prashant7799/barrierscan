"""Synthetic OHLC series for tests. Random walks only: no real data."""

from __future__ import annotations

import numpy as np


def random_walk_ohlc(
    n_bars: int,
    seed: int,
    *,
    step: float = 1.0,
    wick: float = 1.0,
    gap: float = 0.0,
    tick: float | None = None,
    start: float = 100.0,
):
    """Return (open, high, low, close) for a random walk.

    step  size of the close-to-close moves
    wick  how far highs and lows reach past the bar body
    gap   size of jumps between one close and the next open (0 = no gaps)
    tick  if set, round every price to a multiple of tick; this makes exact
          barrier touches (price == barrier) common, which tests the edges
    """
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0.0, step, n_bars))
    open_ = np.empty(n_bars)
    open_[0] = start
    open_[1:] = close[:-1]
    if gap > 0:
        open_[1:] += rng.normal(0.0, gap, n_bars - 1)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, wick, n_bars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, wick, n_bars))

    if tick is not None:
        open_ = np.round(open_ / tick) * tick
        close = np.round(close / tick) * tick
        # Round highs up and lows down so the bar still contains open and close.
        high = np.maximum(np.ceil(high / tick) * tick, np.maximum(open_, close))
        low = np.minimum(np.floor(low / tick) * tick, np.minimum(open_, close))
    return open_, high, low, close


def flat_bars(prices):
    """Bars where open = high = low = close: a close-only series as OHLC."""
    prices = np.asarray(prices, dtype=np.float64)
    return prices.copy(), prices.copy(), prices.copy(), prices.copy()
