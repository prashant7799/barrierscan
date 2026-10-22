"""Bad inputs must fail loudly with a clear message."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import random_walk_ohlc

from barrierscan import brute_force_one, first_touch

OHLC = random_walk_ohlc(50, seed=0)
GOOD = {"tp": 2.0, "sl": 2.0, "horizon": 5}


def call(ohlc=OHLC, **overrides):
    kwargs = {**GOOD, **overrides}
    return first_touch(*ohlc, **kwargs)


def test_good_inputs_work():
    assert len(call()) == 49


def test_lengths_must_match():
    o, h, low, c = OHLC
    with pytest.raises(ValueError, match="same length"):
        call((o, h, low[:-1], c))


def test_need_two_bars():
    with pytest.raises(ValueError, match="at least 2 bars"):
        call(([1.0], [1.0], [1.0], [1.0]))


def test_prices_must_be_finite():
    o, h, low, c = (array.copy() for array in OHLC)
    h[10] = np.nan
    with pytest.raises(ValueError, match="high must be finite; first bad bar is 10"):
        call((o, h, low, c))


def test_prices_must_be_1d():
    o, h, low, c = OHLC
    with pytest.raises(ValueError, match="1-D"):
        call((o.reshape(5, 10), h, low, c))


def test_bars_must_be_consistent():
    o, h, low, c = (array.copy() for array in OHLC)
    h[7] = min(o[7], c[7]) - 0.1
    with pytest.raises(ValueError, match="high must be >= max"):
        call((o, h, low, c))
    o, h, low, c = (array.copy() for array in OHLC)
    low[8] = max(o[8], c[8]) + 0.1
    with pytest.raises(ValueError, match="low must be <= min"):
        call((o, h, low, c))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_barriers_must_be_positive(bad):
    with pytest.raises(ValueError, match="tp must be finite and > 0"):
        call(tp=bad)
    with pytest.raises(ValueError, match="sl must be finite and > 0"):
        call(sl=bad)


def test_barriers_are_only_checked_at_events():
    tp = np.full(50, 2.0)
    tp[:10] = np.nan  # a warm-up period you skip with events
    assert len(call(tp=tp, events=np.arange(10, 49))) == 39
    with pytest.raises(ValueError, match="bad at bar 9"):
        call(tp=tp, events=np.arange(9, 49))


def test_barrier_array_length():
    with pytest.raises(ValueError, match="one value per bar"):
        call(tp=np.ones(49))


@pytest.mark.parametrize("bad", [0, -3])
def test_horizon_must_be_at_least_one(bad):
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        call(horizon=bad)


@pytest.mark.parametrize("bad", [2.5, "5", True, None])
def test_horizon_must_be_an_int(bad):
    with pytest.raises(TypeError, match="horizon must be an int"):
        call(horizon=bad)


def test_numpy_int_horizon_is_fine():
    assert len(call(horizon=np.int32(5))) == 49


def test_entry_must_be_known():
    with pytest.raises(ValueError, match="entry must be"):
        call(entry="open")
    with pytest.raises(ValueError, match="custom entry must be an array"):
        call(entry=100.0)
    with pytest.raises(ValueError, match="one value per bar"):
        call(entry=np.ones(10))
    bad_prices = OHLC[3].copy()
    bad_prices[4] = np.inf
    with pytest.raises(ValueError, match="entry prices must be finite"):
        call(entry=bad_prices)


def test_tie_must_be_known():
    with pytest.raises(ValueError, match="tie must be one of"):
        call(tie="optimistic")


def test_gap_exit_must_be_bool():
    with pytest.raises(TypeError, match="gap_exit"):
        call(gap_exit="yes")


def test_entry_offset_must_be_non_negative():
    with pytest.raises(ValueError, match="entry_offset"):
        call(entry_offset=-0.1)


@pytest.mark.parametrize(
    ("events", "error", "message"),
    [
        ([49], ValueError, "between 0 and 48"),
        ([-1, 3], ValueError, "between 0 and 48"),
        ([5, 3], ValueError, "strictly increasing"),
        ([3, 3], ValueError, "strictly increasing"),
        ([1.0, 2.0], TypeError, "integer"),
        ([[1, 2]], ValueError, "1-D"),
    ],
)
def test_events_must_be_valid(events, error, message):
    with pytest.raises(error, match=message):
        call(events=events)


def test_empty_events_give_an_empty_result():
    res = call(events=[])
    assert len(res) == 0 and res.y.dtype == np.int8


def test_oracle_checks_inputs_too():
    with pytest.raises(ValueError, match="between 0 and 48"):
        brute_force_one(*OHLC, 49, **GOOD)
    with pytest.raises(TypeError, match="integer"):
        brute_force_one(*OHLC, 2.0, **GOOD)
    with pytest.raises(ValueError, match="tie must be one of"):
        brute_force_one(*OHLC, 2, tie="maybe", **GOOD)
