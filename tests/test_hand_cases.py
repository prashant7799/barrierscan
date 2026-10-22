"""Small cases worked out by hand. Each one pins down one rule."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import random_walk_ohlc

from barrierscan import brute_force_one, first_touch


def bars(rows):
    """rows of [open, high, low, close] -> four arrays."""
    table = np.array(rows, dtype=np.float64)
    return table[:, 0], table[:, 1], table[:, 2], table[:, 3]


def label(rows, **kwargs):
    return first_touch(*bars(rows), **kwargs)


def test_clean_long_win():
    # Entry at open[1] = 100. Bar 1 rises to 160 without touching the 90 stop.
    rows = [[100, 100, 100, 100], [100, 160, 95, 150], [150, 150, 150, 150]]
    res = label(rows, tp=50, sl=10, horizon=5)
    assert res.y[0] == 1 and res.t1[0] == 1 and not res.unresolved[0]
    assert res.long_exit_price[0] == 150.0 and res.long_ret_in_r[0] == 5.0


def test_short_win_is_the_mirror_case():
    rows = [[100, 100, 100, 100], [100, 105, 40, 45], [45, 45, 45, 45]]
    res = label(rows, tp=50, sl=10, horizon=5)
    assert res.y[0] == -1 and res.short_exit_price[0] == 50.0 and res.short_ret_in_r[0] == 5.0


def test_one_bar_touching_both_barriers_follows_the_tie_rule():
    # Bar 1 spans 88..155. Long: target 150 and stop 90 both touched.
    # Short: stop 110 touched, target 50 not touched.
    rows = [[100, 100, 100, 100], [100, 155, 88, 120], [120, 120, 120, 120]]
    sl_first = label(rows, tp=50, sl=10, horizon=5, tie="sl_first")
    tp_first = label(rows, tp=50, sl=10, horizon=5, tie="tp_first")
    # sl_first: both sides stopped on bar 1, so y = 0 and the label is resolved.
    assert sl_first.y[0] == 0 and not sl_first.unresolved[0] and sl_first.t1[0] == 1
    assert sl_first.long_outcome[0] == -1 and sl_first.short_outcome[0] == -1
    # tp_first: the long side counts its target.
    assert tp_first.y[0] == 1 and tp_first.long_outcome[0] == 1


def test_entry_convention_moves_the_barriers():
    # close[0] = 90 and open[1] = 100, so the +50 target is 140 or 150.
    rows = [[90, 92, 88, 90], [100, 145, 99, 140], [140, 141, 139, 140]]
    at_close = label(rows, tp=50, sl=10, horizon=5, entry="close")
    at_next_open = label(rows, tp=50, sl=10, horizon=5, entry="next_open")
    assert at_close.y[0] == 1 and at_close.entry_price[0] == 90.0
    assert at_next_open.y[0] == 0 and at_next_open.entry_price[0] == 100.0
    assert at_next_open.unresolved[0]  # the long side timed out


def test_custom_entry_prices():
    o, h, low, c = random_walk_ohlc(200, seed=3)
    by_name = first_touch(o, h, low, c, tp=2, sl=2, horizon=10, entry="close")
    by_array = first_touch(o, h, low, c, tp=2, sl=2, horizon=10, entry=c)
    assert np.array_equal(by_name.y, by_array.y)
    assert np.array_equal(by_name.long_exit_price, by_array.long_exit_price)

    next_open = np.append(o[1:], np.nan)  # the last value is never used
    by_name = first_touch(o, h, low, c, tp=2, sl=2, horizon=10, entry="next_open")
    by_array = first_touch(o, h, low, c, tp=2, sl=2, horizon=10, entry=next_open)
    assert np.array_equal(by_name.y, by_array.y)
    assert np.array_equal(by_name.t1, by_array.t1)


def test_horizon_cap_gives_unresolved_zero():
    rows = [[100, 100.5, 99.5, 100]] * 50  # flat: nothing is ever touched
    res = label(rows, tp=50, sl=10, horizon=10)
    assert np.all(res.y == 0) and np.all(res.unresolved)
    assert res.t1[0] == 10  # the scan covered bars 1..10
    assert res.long_outcome[0] == 0 and res.long_exit_price[0] == 100.0  # close of bar 10


def test_end_of_data_gives_unresolved_zero():
    rows = [[100, 100.5, 99.5, 100]] * 5
    res = label(rows, tp=50, sl=10, horizon=10)
    assert res.t1[3] == 4 and res.unresolved[3]  # only bar 4 was left to scan


def test_last_bar_is_not_an_event_by_default():
    o, h, low, c = random_walk_ohlc(100, seed=0)
    res = first_touch(o, h, low, c, tp=3, sl=3, horizon=20)
    assert len(res) == 99 and np.array_equal(res.index, np.arange(99))


def test_per_side_outcomes():
    rows = [
        [100, 100, 100, 100],
        [100, 104, 89, 95],  # long stop (90) touched; short barriers untouched
        [95, 96, 80, 82],
        [82, 83, 49, 50],  # short target (50) touched
    ]
    res = label(rows, tp=50, sl=10, horizon=5)
    assert res.y[0] == -1 and res.t1[0] == 3
    assert res.long_outcome[0] == -1 and res.long_t1[0] == 1
    assert res.short_outcome[0] == 1 and res.short_t1[0] == 3


def test_each_side_runs_to_its_own_end():
    # tp < sl. The long target is hit on bar 1; the short target only on bar 3.
    # y is decided on bar 1, but the short side keeps going until bar 3.
    rows = [
        [100, 100, 100, 100],
        [100, 101.5, 99.5, 101],  # long target 101 touched
        [101, 101, 99.5, 99.5],
        [99.5, 99.5, 98.5, 99],  # short target 99 touched
    ]
    res = label(rows, tp=1, sl=3, horizon=5)
    assert res.y[0] == 1 and res.t1[0] == 1
    assert res.short_outcome[0] == 1 and res.short_t1[0] == 3


def test_narrow_target_both_sides_win_on_one_wide_bar():
    # tp < sl: one wide bar touches both targets (102 and 98) but no stop
    # (95 and 105). Both sides win on the same bar, so y = 0, under either rule.
    rows = [[100, 100, 100, 100], [100, 102.5, 97.5, 100], [100, 100, 100, 100]]
    for tie in ("sl_first", "tp_first"):
        res = label(rows, tp=2, sl=5, horizon=5, tie=tie)
        assert res.long_outcome[0] == 1 and res.short_outcome[0] == 1
        assert res.y[0] == 0 and not res.unresolved[0] and res.t1[0] == 1
        assert np.isnan(res.exit_price[0]) and np.isnan(res.ret_in_r[0])


def test_gap_exit_fills_at_the_open():
    # close[0] = 100. Bar 1 gaps down and opens at 90, past the long stop (95)
    # and past the short target (95).
    rows = [[100, 100, 100, 100], [90, 91, 85, 88], [88, 88, 88, 88]]
    plain = label(rows, tp=5, sl=5, horizon=5, entry="close", gap_exit=False)
    gaps = label(rows, tp=5, sl=5, horizon=5, entry="close", gap_exit=True)
    assert plain.y[0] == -1 and gaps.y[0] == -1
    assert plain.long_exit_price[0] == 95.0 and plain.long_ret_in_r[0] == -1.0
    assert gaps.long_exit_price[0] == 90.0 and gaps.long_ret_in_r[0] == -2.0
    assert plain.short_exit_price[0] == 95.0 and plain.short_ret_in_r[0] == 1.0
    assert gaps.short_exit_price[0] == 90.0 and gaps.short_ret_in_r[0] == 2.0


def test_gap_exit_settles_a_same_bar_tie():
    # Bar 1 opens at 106, past the long target (105), then falls to 94.
    rows = [[100, 100, 100, 100], [106, 107, 94, 95], [95, 95, 95, 95]]
    plain = label(rows, tp=5, sl=5, horizon=5, entry="close", tie="sl_first")
    gaps = label(rows, tp=5, sl=5, horizon=5, entry="close", tie="sl_first", gap_exit=True)
    assert plain.long_outcome[0] == -1  # tie -> stop
    assert gaps.long_outcome[0] == 1 and gaps.long_exit_price[0] == 106.0


def test_entry_offset_moves_fills_against_you():
    # Reference 100, offset 1: the long fills at 101, so its target is 106.
    rows = [[100, 100, 100, 100], [100, 105.5, 99, 105], [105, 105, 105, 105]]
    no_cost = label(rows, tp=5, sl=3, horizon=1, entry="close")
    with_cost = label(rows, tp=5, sl=3, horizon=1, entry="close", entry_offset=1.0)
    assert no_cost.long_outcome[0] == 1
    assert with_cost.long_outcome[0] == 0  # 105.5 < 106: timed out
    assert with_cost.long_exit_price[0] == 105.0
    assert with_cost.long_ret_in_r[0] == pytest.approx((105.0 - 101.0) / 3.0)


def test_per_bar_barriers_and_events():
    o, h, low, c = random_walk_ohlc(300, seed=5)
    tp = np.linspace(1.0, 4.0, 300)
    sl = np.full(300, 2.0)
    events = np.array([3, 50, 51, 200, 298])
    res = first_touch(o, h, low, c, tp=tp, sl=sl, horizon=15, events=events)
    assert np.array_equal(res.index, events)
    for position, bar in enumerate(events):
        alone = first_touch(o, h, low, c, tp=tp[bar], sl=2.0, horizon=15, events=[bar])
        assert alone.y[0] == res.y[position] and alone.t1[0] == res.t1[position]


def test_t1_and_excursions_stay_inside_the_scan_window():
    o, h, low, c = random_walk_ohlc(500, seed=9)
    res = first_touch(o, h, low, c, tp=3, sl=2, horizon=30)
    last_bar = np.minimum(res.index + 30, 499)
    assert np.all(res.t1 >= res.index + 1) and np.all(res.t1 <= last_bar)
    assert np.all(res.long_t1 <= last_bar) and np.all(res.short_t1 <= last_bar)
    for position, bar in enumerate(res.index):
        window = slice(bar + 1, res.t1[position] + 1)
        assert res.up_excursion[position] == h[window].max() - res.entry_price[position]
        assert res.down_excursion[position] == res.entry_price[position] - low[window].min()


def test_oracle_is_public_and_matches_a_hand_case():
    rows = [[100, 100, 100, 100], [100, 104, 89, 95], [95, 96, 80, 82], [82, 83, 49, 50]]
    row = brute_force_one(*bars(rows), 0, tp=50, sl=10, horizon=5)
    assert row["y"] == -1 and row["t1"] == 3 and row["long_t1"] == 1
    assert row["exit_price"] == 50.0 and row["ret_in_r"] == 5.0
