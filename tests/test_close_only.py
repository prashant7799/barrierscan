"""Cross-check against a second, close-only oracle.

`close_only_first_passage` below is written a different way: it builds a
matrix of future price moves and finds the first crossing with argmax,
with no bar-by-bar loop. On flat bars (open = high = low = close) the
OHLC labeller must agree with it exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import flat_bars, random_walk_ohlc

from barrierscan import first_touch


def close_only_first_passage(prices, tp, sl, horizon):
    """Close-only labels with entry at prices[i]. Returns a dict of arrays.

    For the event at bar i, look at prices[i+1 .. i+horizon] (cut at the end
    of the data). A side ends at the first price at or past its target or stop.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n_bars = prices.shape[0]
    n_events = n_bars - 1

    # moves[i, k-1] = prices[i + k] - prices[i]; NaN past the end of the data.
    moves = np.full((n_events, horizon), np.nan)
    for k in range(1, horizon + 1):
        moves[: n_bars - k, k - 1] = prices[k:] - prices[: n_bars - k]
    last_step = np.minimum(np.arange(n_events) + horizon, n_bars - 1) - np.arange(n_events)

    def first_step(hit):
        """1-based step of the first True in each row, or 0 if none."""
        has_hit = hit.any(axis=1)
        return np.where(has_hit, hit.argmax(axis=1) + 1, 0)

    def side(target_hit, stop_hit):
        target_step = first_step(target_hit)
        stop_step = first_step(stop_hit)
        never = horizon + 1
        target_step = np.where(target_step == 0, never, target_step)
        stop_step = np.where(stop_step == 0, never, stop_step)
        outcome = np.where(target_step < stop_step, 1, np.where(stop_step < target_step, -1, 0))
        end_step = np.where(outcome == 0, last_step, np.minimum(target_step, stop_step))
        return outcome, np.arange(n_events) + end_step

    with np.errstate(invalid="ignore"):  # comparisons with NaN are False, as wanted
        long_outcome, long_t1 = side(moves >= tp, moves <= -sl)
        short_outcome, short_t1 = side(moves <= -tp, moves >= sl)

    y = np.zeros(n_events, dtype=np.int64)
    long_first = (long_outcome == 1) & ((short_outcome != 1) | (long_t1 < short_t1))
    short_first = (short_outcome == 1) & ((long_outcome != 1) | (short_t1 < long_t1))
    y[long_first] = 1
    y[short_first] = -1
    # The label ends at the winner's bar, or when both sides have ended.
    t1 = np.maximum(long_t1, short_t1)
    t1[long_first] = long_t1[long_first]
    t1[short_first] = short_t1[short_first]
    return {
        "y": y,
        "t1": t1,
        "long_outcome": long_outcome,
        "long_t1": long_t1,
        "short_outcome": short_outcome,
        "short_t1": short_t1,
    }


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(("tp", "sl"), [(2.0, 1.0), (1.5, 1.5), (1.0, 2.5)])
@pytest.mark.parametrize("horizon", [1, 7, 40])
def test_flat_bars_match_the_close_only_oracle(seed, tp, sl, horizon):
    _, _, _, close = random_walk_ohlc(300, seed, tick=0.5)
    result = first_touch(*flat_bars(close), tp=tp, sl=sl, horizon=horizon, entry="close")
    expected = close_only_first_passage(close, tp, sl, horizon)
    for key, values in expected.items():
        assert np.array_equal(getattr(result, key), values), key


@pytest.mark.parametrize("seed", range(10))
def test_ohlc_never_resolves_later_than_close_only(seed):
    """Highs and lows reach a barrier no later than closes do."""
    open_, high, low, close = random_walk_ohlc(400, seed, gap=0.5)
    result = first_touch(open_, high, low, close, tp=2.0, sl=1.5, horizon=30, entry="close")
    closes_only = close_only_first_passage(close, 2.0, 1.5, 30)
    for side in ("long", "short"):
        resolved = closes_only[f"{side}_outcome"] != 0
        ohlc_end = getattr(result, f"{side}_t1")[resolved]
        assert np.all(ohlc_end <= closes_only[f"{side}_t1"][resolved])
        # And a side the closes resolve is never a timeout on OHLC bars.
        assert np.all(getattr(result, f"{side}_outcome")[resolved] != 0)
