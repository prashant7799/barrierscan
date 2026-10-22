"""Synthetic OHLC bars for the examples: a random walk, no real data.

A random walk has no edge by construction, so nothing in these examples
says anything about real markets.
"""

import numpy as np


def random_walk_ohlc(n_bars, seed, step=1.0, wick=1.0, start=100.0):
    """Return (open, high, low, close) arrays for a random walk.

    step  typical size of a close-to-close move
    wick  typical distance from the bar body to its high or low
    start first open price
    """
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0.0, step, n_bars))
    open_ = np.empty(n_bars)
    open_[0] = start
    open_[1:] = close[:-1]  # each bar opens at the previous close
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, wick, n_bars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, wick, n_bars))
    return open_, high, low, close
