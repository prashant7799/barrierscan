"""Hypothesis strategies shared by the property tests.

They build small random OHLC series on a 0.5 price grid, so exact barrier
touches and same-bar ties happen often. Every price is a multiple of 0.5,
so adding or doubling prices is exact in floating point.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

GRID = 0.5


@st.composite
def ohlc_series(draw):
    """Random valid bars: low <= open, close <= high, with gaps between bars."""
    n_bars = draw(st.integers(min_value=2, max_value=30))
    moves = draw(st.lists(st.integers(-4, 4), min_size=n_bars, max_size=n_bars))
    gaps = draw(st.lists(st.integers(-2, 2), min_size=n_bars, max_size=n_bars))
    wicks_up = draw(st.lists(st.integers(0, 3), min_size=n_bars, max_size=n_bars))
    wicks_down = draw(st.lists(st.integers(0, 3), min_size=n_bars, max_size=n_bars))

    opens, highs, lows, closes = [], [], [], []
    previous_close = 200
    for k in range(n_bars):
        bar_open = previous_close + gaps[k]
        bar_close = bar_open + moves[k]
        opens.append(bar_open)
        closes.append(bar_close)
        highs.append(max(bar_open, bar_close) + wicks_up[k])
        lows.append(min(bar_open, bar_close) - wicks_down[k])
        previous_close = bar_close
    return tuple(
        np.array(column, dtype=np.float64) * GRID for column in (opens, highs, lows, closes)
    )


@st.composite
def labelling_case(draw, custom_entry: bool = False):
    """Bars plus keyword arguments for `first_touch`.

    With custom_entry=True, some cases use an array of entry prices (the
    close moved by a few grid steps) instead of "next_open" or "close".
    """
    bars = draw(ohlc_series())
    n_bars = bars[0].shape[0]
    distance = st.integers(1, 8).map(lambda ticks: ticks * GRID)
    if draw(st.booleans()):
        tp = draw(distance)
        sl = draw(distance)
    else:
        tp = np.array(draw(st.lists(distance, min_size=n_bars, max_size=n_bars)))
        sl = np.array(draw(st.lists(distance, min_size=n_bars, max_size=n_bars)))

    entry_modes = ["next_open", "close", "custom"] if custom_entry else ["next_open", "close"]
    entry = draw(st.sampled_from(entry_modes))
    if entry == "custom":
        steps = draw(st.lists(st.integers(-3, 3), min_size=n_bars, max_size=n_bars))
        entry = bars[3] + np.array(steps, dtype=np.float64) * GRID

    kwargs = {
        "tp": tp,
        "sl": sl,
        "horizon": draw(st.integers(1, 15)),
        "entry": entry,
        "tie": draw(st.sampled_from(["sl_first", "tp_first"])),
        "gap_exit": draw(st.booleans()),
        "entry_offset": draw(st.sampled_from([0.0, 0.0, GRID])),
    }
    return bars, kwargs
