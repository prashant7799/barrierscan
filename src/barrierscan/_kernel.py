"""The fast labelling loop, compiled with numba.

The functions here take plain arrays that `first_touch` has already
checked. `oracle.py` applies the same rules in slow, plain Python; the
tests compare the two on many random series.

Side outcomes use these codes:
    +1  the side reached its target (take-profit) first
    -1  the side reached its stop (stop-loss) first
     0  neither, before the scan window ended (a timeout)
"""

import numpy as np
from numba import njit

TARGET = 1
STOP = -1
TIMEOUT = 0


@njit(cache=True)
def scan_side(
    open_, high, low, close, first_bar, last_bar, fill, tp, sl, is_long, tp_first, gap_exit
):
    """Scan bars first_bar..last_bar for one side of one event.

    Returns (outcome, bar where it ended, exit price).
    """
    if is_long:
        target = fill + tp
        stop = fill - sl
    else:
        target = fill - tp
        stop = fill + sl

    for j in range(first_bar, last_bar + 1):
        if is_long:
            opened_past_target = open_[j] >= target
            opened_past_stop = open_[j] <= stop
            touched_target = high[j] >= target
            touched_stop = low[j] <= stop
        else:
            opened_past_target = open_[j] <= target
            opened_past_stop = open_[j] >= stop
            touched_target = low[j] <= target
            touched_stop = high[j] >= stop

        # Gap-aware exits: a bar that opens past a barrier fills at its open,
        # and that barrier was clearly reached first.
        if gap_exit and opened_past_target:
            return TARGET, j, open_[j]
        if gap_exit and opened_past_stop:
            return STOP, j, open_[j]

        if touched_target and touched_stop:
            # One bar holds both barriers. Bars do not show which came first,
            # so the tie rule decides.
            if tp_first:
                return TARGET, j, target
            return STOP, j, stop
        if touched_target:
            return TARGET, j, target
        if touched_stop:
            return STOP, j, stop

    # Time ran out: exit at the close of the last bar scanned.
    return TIMEOUT, last_bar, close[last_bar]


@njit(cache=True)
def label_events(
    open_, high, low, close, events, ref_price, tp, sl, offset, horizon, tp_first, gap_exit
):
    """Label every event. All per-event inputs have one value per event."""
    n_bars = close.shape[0]
    n_events = events.shape[0]

    y = np.zeros(n_events, np.int8)
    t1 = np.zeros(n_events, np.int64)
    unresolved = np.zeros(n_events, np.bool_)
    up_excursion = np.zeros(n_events, np.float64)
    down_excursion = np.zeros(n_events, np.float64)
    long_outcome = np.zeros(n_events, np.int8)
    long_t1 = np.zeros(n_events, np.int64)
    long_exit_price = np.zeros(n_events, np.float64)
    long_ret_in_r = np.zeros(n_events, np.float64)
    short_outcome = np.zeros(n_events, np.int8)
    short_t1 = np.zeros(n_events, np.int64)
    short_exit_price = np.zeros(n_events, np.float64)
    short_ret_in_r = np.zeros(n_events, np.float64)

    for k in range(n_events):
        i = events[k]
        first_bar = i + 1
        last_bar = min(i + horizon, n_bars - 1)

        # A long buys above the reference price, a short sells below it.
        long_fill = ref_price[k] + offset[k]
        short_fill = ref_price[k] - offset[k]

        long_out, long_bar, long_exit = scan_side(
            open_,
            high,
            low,
            close,
            first_bar,
            last_bar,
            long_fill,
            tp[k],
            sl[k],
            True,
            tp_first,
            gap_exit,
        )
        short_out, short_bar, short_exit = scan_side(
            open_,
            high,
            low,
            close,
            first_bar,
            last_bar,
            short_fill,
            tp[k],
            sl[k],
            False,
            tp_first,
            gap_exit,
        )

        # Combine the two sides into one label.
        long_won = long_out == TARGET
        short_won = short_out == TARGET
        if long_won and short_won and long_bar == short_bar:
            # Both targets on the same bar: no clear winner.
            label = 0
            end_bar = long_bar
        elif long_won and (not short_won or long_bar < short_bar):
            label = 1
            end_bar = long_bar
        elif short_won:
            label = -1
            end_bar = short_bar
        else:
            # No target was reached. The label is known once both sides end.
            label = 0
            end_bar = max(long_bar, short_bar)
            unresolved[k] = long_out == TIMEOUT or short_out == TIMEOUT

        # Biggest move up and down while waiting for the label.
        highest = high[first_bar]
        lowest = low[first_bar]
        for j in range(first_bar + 1, end_bar + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]

        y[k] = label
        t1[k] = end_bar
        up_excursion[k] = highest - ref_price[k]
        down_excursion[k] = ref_price[k] - lowest

        long_outcome[k] = long_out
        long_t1[k] = long_bar
        long_exit_price[k] = long_exit
        long_ret_in_r[k] = (long_exit - long_fill) / sl[k]

        short_outcome[k] = short_out
        short_t1[k] = short_bar
        short_exit_price[k] = short_exit
        short_ret_in_r[k] = (short_fill - short_exit) / sl[k]

    return (
        y,
        t1,
        unresolved,
        up_excursion,
        down_excursion,
        long_outcome,
        long_t1,
        long_exit_price,
        long_ret_in_r,
        short_outcome,
        short_t1,
        short_exit_price,
        short_ret_in_r,
    )
