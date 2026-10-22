"""Adversarial tests: odd inputs, extreme settings and symmetry rules.

The property tests here change the input in a way that must not change the
labels (dropping earlier bars, cutting bars after the scan window, moving
or scaling every price) and check that the labels really stay the same.
Every price is on a 0.5 grid and every change is exact in floating point,
so these tests compare with ==, not with a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import GRID, labelling_case
from synthetic import random_walk_ohlc

from barrierscan import brute_force_one, first_touch

SETTINGS = settings(max_examples=200, deadline=None)

# Result fields that hold bar positions, and fields that hold everything else.
POSITION_FIELDS = ("index", "t1", "long_t1", "short_t1")
OTHER_FIELDS = (
    "y",
    "unresolved",
    "entry_price",
    "up_excursion",
    "down_excursion",
    "long_outcome",
    "long_exit_price",
    "long_ret_in_r",
    "short_outcome",
    "short_exit_price",
    "short_ret_in_r",
)
PER_BAR_KWARGS = ("tp", "sl", "entry", "entry_offset")


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


def cut(bars, kwargs, start, stop):
    """Keep bars start..stop-1, and the matching part of any per-bar argument."""
    new_bars = tuple(column[start:stop] for column in bars)
    new_kwargs = dict(kwargs)
    for name in PER_BAR_KWARGS:
        value = kwargs[name]
        if isinstance(value, np.ndarray):
            new_kwargs[name] = value[start:stop]
    return new_bars, new_kwargs


# ---------------------------------------------------------------------------
# Changes that must not change the labels
# ---------------------------------------------------------------------------


@SETTINGS
@given(labelling_case(custom_entry=True), st.data())
def test_dropping_earlier_bars_only_shifts_bar_positions(case, data):
    """A label depends on its own bar and later bars, never on earlier ones."""
    bars, kwargs = case
    n_bars = bars[0].shape[0]
    start = data.draw(st.integers(0, n_bars - 2), label="start")

    full = first_touch(*bars, **kwargs)
    later_bars, later_kwargs = cut(bars, kwargs, start, n_bars)
    later = first_touch(*later_bars, **later_kwargs)
    keep = full.index >= start

    for name in POSITION_FIELDS:
        assert np.array_equal(getattr(later, name) + start, getattr(full, name)[keep]), name
    for name in OTHER_FIELDS:
        assert same(getattr(later, name), getattr(full, name)[keep]), name


@SETTINGS
@given(labelling_case(custom_entry=True), st.data())
def test_bars_after_the_scan_window_do_not_matter(case, data):
    """Cutting off bars after i + horizon must not change the row of event i."""
    bars, kwargs = case
    n_bars = bars[0].shape[0]
    stop = data.draw(st.integers(2, n_bars), label="stop")

    full = first_touch(*bars, **kwargs)
    shorter_bars, shorter_kwargs = cut(bars, kwargs, 0, stop)
    shorter = first_touch(*shorter_bars, **shorter_kwargs)
    # Rows whose whole scan window fits inside the shorter data.
    fits = shorter.index + kwargs["horizon"] <= stop - 1

    for name in POSITION_FIELDS + OTHER_FIELDS:
        assert same(getattr(shorter, name)[fits], getattr(full, name)[: stop - 1][fits]), name


@SETTINGS
@given(labelling_case(custom_entry=True), st.integers(-400, 400))
def test_moving_every_price_moves_only_the_prices(case, steps):
    bars, kwargs = case
    shift = steps * GRID
    moved_bars = tuple(column + shift for column in bars)
    moved_kwargs = dict(kwargs)
    if isinstance(kwargs["entry"], np.ndarray):
        moved_kwargs["entry"] = kwargs["entry"] + shift

    before = first_touch(*bars, **kwargs)
    after = first_touch(*moved_bars, **moved_kwargs)

    for name in ("entry_price", "long_exit_price", "short_exit_price"):
        assert np.array_equal(getattr(after, name), getattr(before, name) + shift), name
    for name in POSITION_FIELDS + OTHER_FIELDS:
        if name not in ("entry_price", "long_exit_price", "short_exit_price"):
            assert same(getattr(after, name), getattr(before, name)), name


@SETTINGS
@given(labelling_case(custom_entry=True), st.sampled_from([0.25, 0.5, 2.0, 4.0]))
def test_scaling_prices_and_distances_together_keeps_the_labels(case, factor):
    """Measuring in other units (cents instead of dollars, say) changes nothing."""
    bars, kwargs = case
    scaled_bars = tuple(column * factor for column in bars)
    scaled_kwargs = dict(kwargs)
    for name in PER_BAR_KWARGS:
        if not isinstance(kwargs[name], str):
            scaled_kwargs[name] = kwargs[name] * factor

    before = first_touch(*bars, **kwargs)
    after = first_touch(*scaled_bars, **scaled_kwargs)

    in_price_units = (
        "entry_price",
        "up_excursion",
        "down_excursion",
        "long_exit_price",
        "short_exit_price",
    )
    for name in in_price_units:
        assert np.array_equal(getattr(after, name), getattr(before, name) * factor), name
    for name in POSITION_FIELDS + OTHER_FIELDS:
        if name not in in_price_units:
            assert same(getattr(after, name), getattr(before, name)), name


# ---------------------------------------------------------------------------
# Rules connecting the fields of one result
# ---------------------------------------------------------------------------


@SETTINGS
@given(labelling_case(custom_entry=True))
def test_y_and_t1_follow_from_the_two_sides(case):
    bars, kwargs = case
    res = first_touch(*bars, **kwargs)

    long_won = res.long_outcome == 1
    short_won = res.short_outcome == 1
    long_first = long_won & (~short_won | (res.long_t1 < res.short_t1))
    short_first = short_won & (~long_won | (res.short_t1 < res.long_t1))
    assert not np.any(long_first & short_first)

    expected_y = np.zeros(len(res), dtype=np.int8)
    expected_y[long_first] = 1
    expected_y[short_first] = -1
    assert np.array_equal(res.y, expected_y)

    # No winner (or a same-bar double win): the label ends when both sides have ended.
    expected_t1 = np.maximum(res.long_t1, res.short_t1)
    expected_t1[long_first] = res.long_t1[long_first]
    expected_t1[short_first] = res.short_t1[short_first]
    assert np.array_equal(res.t1, expected_t1)


@SETTINGS
@given(labelling_case(custom_entry=True))
def test_tie_rule_and_gap_exit_never_move_where_a_side_ends(case):
    """Both settings only decide what happened on the bar where a side ends."""
    bars, kwargs = case
    runs = {
        (tie, gap_exit): first_touch(*bars, **{**kwargs, "tie": tie, "gap_exit": gap_exit})
        for tie in ("sl_first", "tp_first")
        for gap_exit in (False, True)
    }
    reference = runs[("sl_first", False)]
    for run in runs.values():
        assert np.array_equal(run.long_t1, reference.long_t1)
        assert np.array_equal(run.short_t1, reference.short_t1)

    # tp_first can only turn a stop into a target, never the other way round.
    for gap_exit in (False, True):
        cautious = runs[("sl_first", gap_exit)]
        hopeful = runs[("tp_first", gap_exit)]
        assert np.all(hopeful.long_outcome >= cautious.long_outcome)
        assert np.all(hopeful.short_outcome >= cautious.short_outcome)


@SETTINGS
@given(labelling_case(custom_entry=True))
def test_gap_exits_are_never_better_than_the_barrier(case):
    """With gaps, a stop fills at or beyond the stop and a target at or beyond the target."""
    bars, kwargs = case
    kwargs = {**kwargs, "gap_exit": True}
    res = first_touch(*bars, **kwargs)
    n_bars = bars[0].shape[0]
    close = bars[3]
    tp = np.broadcast_to(kwargs["tp"], (n_bars,))[res.index]
    sl = np.broadcast_to(kwargs["sl"], (n_bars,))[res.index]
    last_bar = np.minimum(res.index + kwargs["horizon"], n_bars - 1)

    for side in ("long", "short"):
        outcome = getattr(res, f"{side}_outcome")
        ret = getattr(res, f"{side}_ret_in_r")
        side_t1 = getattr(res, f"{side}_t1")
        exit_price = getattr(res, f"{side}_exit_price")
        stopped, won, timed_out = outcome == -1, outcome == 1, outcome == 0
        assert np.all(ret[stopped] <= -1.0)
        assert np.all(ret[won] >= tp[won] / sl[won])
        assert np.array_equal(side_t1[timed_out], last_bar[timed_out])
        assert np.array_equal(exit_price[timed_out], close[side_t1[timed_out]])


# ---------------------------------------------------------------------------
# Extreme settings and unusual input types
# ---------------------------------------------------------------------------

BARS = random_walk_ohlc(40, seed=12, gap=0.5, tick=0.25)
KWARGS = {"tp": 1.5, "sl": 1.0, "entry": "close", "tie": "sl_first"}


def assert_same_result(a, b):
    for name in POSITION_FIELDS + OTHER_FIELDS:
        assert same(getattr(a, name), getattr(b, name)), name


@pytest.mark.parametrize(
    "horizon",
    [39, 40, 1_000_000, np.int64(2**63 - 1), np.uint64(2**64 - 1), 10**30],
    ids=["n-1", "n", "1e6", "int64 max", "uint64 max", "1e30"],
)
def test_huge_horizon_scans_to_the_end_of_the_data(horizon):
    # Regression test: a horizon near the int64 limit used to overflow
    # inside the kernel and give wrong rows, or raise OverflowError.
    to_the_end = first_touch(*BARS, horizon=39, **KWARGS)
    res = first_touch(*BARS, horizon=horizon, **KWARGS)
    assert_same_result(res, to_the_end)
    assert np.all(res.t1[res.unresolved] == 39)
    slow = brute_force_one(*BARS, 5, horizon=horizon, **KWARGS)
    assert slow["t1"] == res.t1[5] and slow["y"] == res.y[5]


def test_smallest_possible_input():
    # Two bars, one event. Bar 1 touches both narrow targets and neither stop.
    res = first_touch([1.0, 1.0], [1.0, 3.0], [1.0, 0.0], [1.0, 2.0], tp=1, sl=5, horizon=1)
    assert len(res) == 1 and res.index[0] == 0
    assert res.y[0] == 0 and res.t1[0] == 1 and not res.unresolved[0]
    assert res.long_outcome[0] == 1 and res.short_outcome[0] == 1


def test_offset_larger_than_the_stop_stops_at_once():
    # Long fills at 101 with its stop at 100.5, above the next open (100).
    rows = np.array([[100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100]], float)
    bars = tuple(rows[:, k] for k in range(4))
    plain = first_touch(*bars, tp=2, sl=0.5, horizon=2, entry_offset=1.0)
    gaps = first_touch(*bars, tp=2, sl=0.5, horizon=2, entry_offset=1.0, gap_exit=True)
    assert plain.long_outcome[0] == -1 and plain.long_t1[0] == 1
    assert plain.long_ret_in_r[0] == -1.0  # exit at the stop price
    assert gaps.long_ret_in_r[0] == -2.0  # exit at the open, 1.0 below the fill


def as_pandas_reversed(column):
    pd = pytest.importorskip("pandas")
    return pd.Series(column, index=np.arange(len(column))[::-1])


def as_read_only(column):
    column = column.copy()
    column.setflags(write=False)
    return column


INPUT_KINDS = {
    "list": lambda column: column.tolist(),
    "tuple": lambda column: tuple(column.tolist()),
    "float32": lambda column: column.astype(np.float32),
    "big-endian": lambda column: column.astype(">f8"),
    "strided view": lambda column: np.repeat(column, 2)[::2],
    "read-only": as_read_only,
    "pandas, reversed index": as_pandas_reversed,
}


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_input_types_give_the_same_labels(kind):
    convert = INPUT_KINDS[kind]
    # Prices on a 0.25 grid survive float32 and back exactly.
    columns = [convert(column) for column in BARS]
    expected = first_touch(*BARS, horizon=10, **KWARGS)
    assert_same_result(first_touch(*columns, horizon=10, **KWARGS), expected)


def test_whole_number_prices_work():
    bars = tuple(np.round(column).astype(np.int64) for column in BARS)
    res = first_touch(*bars, tp=2, sl=2, horizon=10)
    floats = first_touch(*(column.astype(np.float64) for column in bars), tp=2, sl=2, horizon=10)
    assert_same_result(res, floats)


def test_inputs_are_never_changed():
    o, h, low, c = random_walk_ohlc(60, seed=4, gap=0.5)
    rng = np.random.default_rng(0)
    arguments = {
        "tp": rng.uniform(0.5, 2.0, 60),
        "sl": rng.uniform(0.5, 2.0, 60),
        "entry": c + 0.1,
        "entry_offset": rng.uniform(0.0, 0.2, 60),
        "events": np.arange(0, 59, 3),
    }
    originals = [array.copy() for array in (o, h, low, c, *arguments.values())]
    res = first_touch(o, h, low, c, horizon=8, gap_exit=True, **arguments)
    one = {key: value for key, value in arguments.items() if key != "events"}
    brute_force_one(o, h, low, c, 3, horizon=8, gap_exit=True, **one)
    for before, after in zip(originals, (o, h, low, c, *arguments.values()), strict=True):
        assert np.array_equal(before, after)
    # The result owns its arrays: changing it does not touch the inputs.
    res.entry_price[:] = 0.0
    res.index[:] = 0
    assert np.array_equal(arguments["entry"], originals[6])
    assert np.array_equal(arguments["events"], originals[8])


def test_custom_entry_only_needs_values_at_events():
    o, h, low, c = BARS
    events = np.arange(5, 39, 4)
    sparse_entry = np.full(40, np.nan)
    sparse_entry[events] = c[events]
    sparse = first_touch(
        o, h, low, c, tp=1.5, sl=1.0, horizon=10, entry=sparse_entry, events=events
    )
    by_name = first_touch(o, h, low, c, tp=1.5, sl=1.0, horizon=10, entry="close", events=events)
    assert_same_result(sparse, by_name)


def test_same_input_gives_the_same_result_twice():
    assert_same_result(
        first_touch(*BARS, horizon=10, **KWARGS), first_touch(*BARS, horizon=10, **KWARGS)
    )


def test_a_true_false_mask_as_events_gets_a_helpful_error():
    mask = BARS[3][:-1] > 100.0
    with pytest.raises(TypeError, match="flatnonzero"):
        first_touch(*BARS, horizon=5, events=mask, **KWARGS)
    # The hint works.
    assert len(first_touch(*BARS, horizon=5, events=np.flatnonzero(mask), **KWARGS)) == mask.sum()


def test_complex_numbers_are_rejected():
    o, h, low, c = BARS
    with pytest.raises(TypeError, match="open must hold real numbers"):
        first_touch(o + 0j, h, low, c, horizon=5, **KWARGS)
    with pytest.raises(TypeError, match="tp must hold real numbers"):
        first_touch(o, h, low, c, horizon=5, **{**KWARGS, "tp": 1.5 + 0j})
