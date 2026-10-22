"""Tests for the helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import random_walk_ohlc

from barrierscan import (
    average_true_range,
    breakeven_hit_rate,
    entry_offset,
    first_touch,
    vol_barriers,
)


def slow_atr(high, low, close, window):
    """ATR the long way, for comparison."""
    true_range = []
    for i in range(len(high)):
        if i == 0:
            true_range.append(high[0] - low[0])
        else:
            true_range.append(
                max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            )
    atr = []
    for i in range(len(high)):
        start = max(0, i - window + 1)
        atr.append(sum(true_range[start : i + 1]) / (i + 1 - start))
    return np.array(atr)


@pytest.mark.parametrize("window", [1, 5, 14, 500])
def test_atr_matches_the_long_way(window):
    _, high, low, close = random_walk_ohlc(300, seed=4, gap=0.5)
    assert np.allclose(
        average_true_range(high, low, close, window), slow_atr(high, low, close, window)
    )


def test_atr_uses_only_past_bars():
    _, high, low, close = random_walk_ohlc(200, seed=8)
    before = average_true_range(high, low, close, 14)
    high2, low2 = high.copy(), low.copy()
    high2[150:] += 50.0  # change the future only
    low2[150:] -= 50.0
    after = average_true_range(high2, low2, close, 14)
    assert np.array_equal(before[:150], after[:150])


def test_atr_rejects_bad_input():
    with pytest.raises(ValueError, match="window"):
        average_true_range([2.0], [1.0], [1.5], window=0)
    with pytest.raises(ValueError, match="high must be >= low"):
        average_true_range([1.0], [2.0], [1.5])
    with pytest.raises(ValueError, match="same length"):
        average_true_range([2.0, 2.0], [1.0], [1.5])


def test_vol_barriers_keep_the_ratio_and_the_floor():
    _, high, low, close = random_walk_ohlc(300, seed=2)
    tp, sl = vol_barriers(high, low, close, window=10, tp_mult=3.0, sl_mult=1.5, floor=1.0)
    atr = average_true_range(high, low, close, 10)
    assert np.allclose(sl, np.maximum(1.5 * atr, 1.0))
    assert np.allclose(tp / sl, 2.0)
    assert np.all(sl >= 1.0)


def test_vol_barriers_floor_per_bar():
    _, high, low, close = random_walk_ohlc(100, seed=2)
    floor = np.linspace(0.0, 10.0, 100)
    _, sl = vol_barriers(high, low, close, floor=floor)
    assert np.all(sl >= floor)
    with pytest.raises(ValueError, match="floor"):
        vol_barriers(high, low, close, floor=np.ones(5))
    with pytest.raises(ValueError, match="floor"):
        vol_barriers(high, low, close, floor=-1.0)
    with pytest.raises(ValueError, match="tp_mult"):
        vol_barriers(high, low, close, tp_mult=0.0)


def test_vol_barriers_feed_first_touch():
    open_, high, low, close = random_walk_ohlc(500, seed=6)
    tp, sl = vol_barriers(high, low, close, floor=0.5)
    result = first_touch(open_, high, low, close, tp=tp, sl=sl, horizon=20)
    assert len(result) == 499


def test_entry_offset():
    assert entry_offset(0.2) == pytest.approx(0.1)
    assert entry_offset(0.2, slippage=0.05) == pytest.approx(0.15)
    assert isinstance(entry_offset(0.2), float)
    assert np.allclose(entry_offset(np.array([0.0, 0.4]), 0.1), [0.1, 0.3])
    with pytest.raises(ValueError, match="spread"):
        entry_offset(-0.1)
    with pytest.raises(ValueError, match="slippage"):
        entry_offset(0.1, slippage=np.nan)


def test_breakeven_hit_rate():
    assert breakeven_hit_rate(1.0, 1.0) == pytest.approx(0.5)
    assert breakeven_hit_rate(3.0, 1.0) == pytest.approx(0.25)
    assert breakeven_hit_rate(60.0, 20.0, cost=11.0) == pytest.approx(31.0 / 80.0)
    assert np.allclose(breakeven_hit_rate(np.array([1.0, 3.0]), 1.0), [0.5, 0.25])
    with pytest.raises(ValueError, match="tp"):
        breakeven_hit_rate(0.0, 1.0)
    with pytest.raises(ValueError, match="sl"):
        breakeven_hit_rate(1.0, -1.0)


def test_breakeven_rate_is_where_expected_return_is_zero():
    rate = breakeven_hit_rate(2.0, 1.0, cost=0.1)
    assert rate * 2.0 - (1 - rate) * 1.0 - 0.1 == pytest.approx(0.0)
