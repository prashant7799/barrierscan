"""The fast kernel must match the slow oracle on every row of many random series."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from synthetic import random_walk_ohlc

from barrierscan import brute_force_one, first_touch

ENTRIES = ["next_open", "close", "custom"]
TIES = ["sl_first", "tp_first"]
GAP_EXITS = [False, True]
# (base tp, base sl): target wider than stop, equal, and narrower than stop.
SHAPES = {"tp>sl": (3.0, 1.0), "tp=sl": (2.0, 2.0), "tp<sl": (1.0, 3.0)}
SEEDS = range(4)
N_BARS = 250

CONFIGS = list(itertools.product(ENTRIES, TIES, GAP_EXITS, SHAPES))


def assert_same_row(fast: dict, slow: dict, where: str) -> None:
    """Every field must be exactly equal (NaN counts as equal to NaN)."""
    assert fast.keys() == slow.keys()
    for key, slow_value in slow.items():
        fast_value = fast[key]
        if isinstance(slow_value, float) and math.isnan(slow_value):
            assert math.isnan(fast_value), f"{key} differs at {where}: {fast_value} vs NaN"
        else:
            assert fast_value == slow_value, (
                f"{key} differs at {where}: {fast_value} vs {slow_value}"
            )


def make_case(entry: str, shape: str, seed: int):
    """A random series plus random per-bar barriers, offsets and entry prices."""
    rng = np.random.default_rng(1000 + seed)
    # Half the seeds use prices on a 0.25 grid, so exact touches happen often.
    tick = 0.25 if seed % 2 == 0 else None
    open_, high, low, close = random_walk_ohlc(N_BARS, seed, gap=0.8, tick=tick)

    base_tp, base_sl = SHAPES[shape]
    scale = rng.choice([0.5, 1.0, 1.5, 2.0], size=N_BARS)
    kwargs = {
        "tp": base_tp * scale,
        "sl": base_sl * scale,
        "horizon": int(rng.integers(1, 41)),
        "entry_offset": rng.choice([0.0, 0.0, 0.25, 0.5], size=N_BARS),
    }
    if entry == "custom":
        kwargs["entry"] = close + rng.choice([-0.5, 0.0, 0.5], size=N_BARS)
    else:
        kwargs["entry"] = entry
    return (open_, high, low, close), kwargs


@pytest.mark.parametrize(("entry", "tie", "gap_exit", "shape"), CONFIGS)
def test_kernel_matches_oracle_on_every_row(entry, tie, gap_exit, shape):
    for seed in SEEDS:
        bars, kwargs = make_case(entry, shape, seed)
        result = first_touch(*bars, tie=tie, gap_exit=gap_exit, **kwargs)
        assert len(result) == N_BARS - 1
        for position, bar in enumerate(result.index):
            slow = brute_force_one(*bars, int(bar), tie=tie, gap_exit=gap_exit, **kwargs)
            where = f"bar {bar}, seed {seed}, {entry}/{tie}/gap={gap_exit}/{shape}"
            assert_same_row(result.row(position), slow, where)


@pytest.mark.parametrize("seed", range(6))
def test_kernel_matches_oracle_with_scalar_barriers_and_sparse_events(seed):
    rng = np.random.default_rng(seed)
    bars = random_walk_ohlc(400, seed, step=0.7, wick=0.5, gap=0.3)
    events = np.sort(rng.choice(399, size=60, replace=False))
    kwargs = {"tp": 2.0, "sl": 1.5, "horizon": 25, "entry": "next_open", "tie": "sl_first"}
    result = first_touch(*bars, events=events, **kwargs)
    assert np.array_equal(result.index, events)
    for position, bar in enumerate(events):
        slow = brute_force_one(*bars, int(bar), **kwargs)
        assert_same_row(result.row(position), slow, f"bar {bar}, seed {seed}")


def test_random_series_reach_every_case():
    """The random cases above are only useful if they hit the hard cases."""
    counts = {
        "y=+1": 0,
        "y=-1": 0,
        "unresolved": 0,
        "both stopped": 0,
        "both targets same bar": 0,
        "tie rule changed a row": 0,
        "gap exit changed a row": 0,
    }
    for shape, seed in itertools.product(SHAPES, SEEDS):
        bars, kwargs = make_case("close", shape, seed)
        base = first_touch(*bars, tie="sl_first", gap_exit=False, **kwargs)
        other_tie = first_touch(*bars, tie="tp_first", gap_exit=False, **kwargs)
        with_gaps = first_touch(*bars, tie="sl_first", gap_exit=True, **kwargs)
        for result in (base, other_tie, with_gaps):
            counts["y=+1"] += int(np.sum(result.y == 1))
            counts["y=-1"] += int(np.sum(result.y == -1))
            counts["unresolved"] += int(np.sum(result.unresolved))
            stopped = (result.long_outcome == -1) & (result.short_outcome == -1)
            counts["both stopped"] += int(np.sum(stopped & (result.y == 0)))
            same_bar = (
                (result.long_outcome == 1)
                & (result.short_outcome == 1)
                & (result.long_t1 == result.short_t1)
            )
            counts["both targets same bar"] += int(np.sum(same_bar))
        counts["tie rule changed a row"] += int(np.sum(base.long_outcome != other_tie.long_outcome))
        counts["gap exit changed a row"] += int(
            np.sum(base.long_exit_price != with_gaps.long_exit_price)
        )
    missing = [name for name, count in counts.items() if count == 0]
    assert not missing, f"random cases never produced: {missing}"
