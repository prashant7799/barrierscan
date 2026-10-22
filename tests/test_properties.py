"""Property tests: rules that must hold for any valid input.

Hypothesis builds small random OHLC series on a 0.5 price grid (see
strategies.py), so exact barrier touches and same-bar ties happen often.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import labelling_case

from barrierscan import brute_force_one, first_touch

SETTINGS = settings(max_examples=200, deadline=None)


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


@SETTINGS
@given(labelling_case(custom_entry=True))
def test_kernel_equals_oracle(case):
    bars, kwargs = case
    result = first_touch(*bars, **kwargs)
    for position, bar in enumerate(result.index):
        slow = brute_force_one(*bars, int(bar), **kwargs)
        fast = result.row(position)
        for key, value in slow.items():
            assert same(fast[key], value), (key, int(bar), fast[key], value)


@SETTINGS
@given(labelling_case(custom_entry=True))
def test_basic_invariants(case):
    bars, kwargs = case
    result = first_touch(*bars, **kwargs)
    n_bars = bars[0].shape[0]
    last_bar = np.minimum(result.index + kwargs["horizon"], n_bars - 1)
    tp = np.broadcast_to(kwargs["tp"], (n_bars,))[result.index]
    offset = kwargs["entry_offset"]

    assert set(np.unique(result.y)) <= {-1, 0, 1}
    assert np.all((result.t1 > result.index) & (result.t1 <= last_bar))

    long_wins = result.y == 1
    short_wins = result.y == -1
    assert np.all(result.long_outcome[long_wins] == 1)
    assert np.all(result.short_outcome[short_wins] == 1)
    # A winner travelled at least tp (plus the offset) in its direction.
    assert np.all(result.up_excursion[long_wins] >= tp[long_wins] + offset)
    assert np.all(result.down_excursion[short_wins] >= tp[short_wins] + offset)

    # unresolved means: no winner and time ran out.
    assert np.all(result.y[result.unresolved] == 0)
    assert np.all(result.t1[result.unresolved] == last_bar[result.unresolved])
    no_winner = (result.long_outcome != 1) & (result.short_outcome != 1)
    timed_out = (result.long_outcome == 0) | (result.short_outcome == 0)
    assert np.array_equal(result.unresolved, no_winner & timed_out)


@SETTINGS
@given(labelling_case(custom_entry=True))
def test_returns_in_r_without_gaps(case):
    bars, kwargs = case
    kwargs = {**kwargs, "gap_exit": False}
    result = first_touch(*bars, **kwargs)
    n_bars = bars[0].shape[0]
    tp = np.broadcast_to(kwargs["tp"], (n_bars,))[result.index]
    sl = np.broadcast_to(kwargs["sl"], (n_bars,))[result.index]
    for side in ("long", "short"):
        outcome = getattr(result, f"{side}_outcome")
        ret = getattr(result, f"{side}_ret_in_r")
        # Exits are at the barrier, so a win is tp/sl R and a loss is -1 R.
        assert np.allclose(ret[outcome == 1], tp[outcome == 1] / sl[outcome == 1])
        assert np.allclose(ret[outcome == -1], -1.0)


@SETTINGS
@given(labelling_case())
def test_mirror_symmetry(case):
    """Flipping prices upside down swaps the long and short sides exactly."""
    (open_, high, low, close), kwargs = case
    mirrored = (-open_, -low, -high, -close)
    result = first_touch(open_, high, low, close, **kwargs)
    flipped = first_touch(*mirrored, **kwargs)

    assert np.array_equal(flipped.y, -result.y)
    assert np.array_equal(flipped.t1, result.t1)
    assert np.array_equal(flipped.unresolved, result.unresolved)
    assert np.array_equal(flipped.up_excursion, result.down_excursion)
    assert np.array_equal(flipped.down_excursion, result.up_excursion)
    assert np.array_equal(flipped.long_outcome, result.short_outcome)
    assert np.array_equal(flipped.long_t1, result.short_t1)
    assert np.array_equal(flipped.long_exit_price, -result.short_exit_price)
    assert np.array_equal(flipped.long_ret_in_r, result.short_ret_in_r)
    assert np.array_equal(flipped.short_ret_in_r, result.long_ret_in_r)


@SETTINGS
@given(labelling_case(custom_entry=True), st.integers(1, 10))
def test_longer_horizon_keeps_resolved_results(case, extra_bars):
    bars, kwargs = case
    short_scan = first_touch(*bars, **kwargs)
    long_scan = first_touch(*bars, **{**kwargs, "horizon": kwargs["horizon"] + extra_bars})

    resolved = ~short_scan.unresolved
    assert np.array_equal(short_scan.y[resolved], long_scan.y[resolved])
    assert np.array_equal(short_scan.t1[resolved], long_scan.t1[resolved])
    for side in ("long", "short"):
        outcome = getattr(short_scan, f"{side}_outcome")
        ended = outcome != 0
        assert np.array_equal(outcome[ended], getattr(long_scan, f"{side}_outcome")[ended])
        assert np.array_equal(
            getattr(short_scan, f"{side}_t1")[ended], getattr(long_scan, f"{side}_t1")[ended]
        )


@SETTINGS
@given(labelling_case(custom_entry=True))
def test_events_subset_matches_full_run(case):
    bars, kwargs = case
    full = first_touch(*bars, **kwargs)
    every_other = full.index[::2]
    subset = first_touch(*bars, events=every_other, **kwargs)
    assert np.array_equal(subset.y, full.y[::2])
    assert np.array_equal(subset.t1, full.t1[::2])
    assert same(subset.long_exit_price, full.long_exit_price[::2])
