"""`first_touch`: the main entry point."""

from __future__ import annotations

import numpy as np

from barrierscan._kernel import label_events
from barrierscan._validate import check_inputs
from barrierscan.result import BarrierResult


def first_touch(
    open,
    high,
    low,
    close,
    *,
    tp,
    sl,
    horizon: int,
    entry="next_open",
    tie: str = "sl_first",
    events=None,
    entry_offset=0.0,
    gap_exit: bool = False,
) -> BarrierResult:
    """Label each event bar by which barrier the next bars touch first.

    For an event at bar i:

    1. Pick a reference entry price: ``open[i + 1]`` ("next_open"),
       ``close[i]`` ("close"), or ``entry[i]`` when ``entry`` is an array.
    2. A long fills at reference + entry_offset and a short at
       reference - entry_offset.
    3. Long: target = fill + tp, stop = fill - sl.
       Short: target = fill - tp, stop = fill + sl.
    4. Scan bars i+1 .. min(i + horizon, last bar). Each side ends on the
       first bar whose high/low touches one of its barriers (touching
       means reaching the price exactly or going past it).
    5. y = +1 if the long target came first, -1 if the short target came
       first, and 0 otherwise.

    Parameters
    ----------
    open, high, low, close : array-like of float
        One value per bar, same length, finite. Any names or source work;
        only the order matters. low <= open, close <= high on every bar.
    tp, sl : float or array-like of float
        Target and stop distances in price units. A scalar applies to every
        bar; an array gives one value per bar (the value at bar i is used for
        the event at bar i). Must be finite and > 0 at every event.
    horizon : int
        Most bars to scan after the event bar (>= 1). A horizon longer than
        the data scans to the last bar.
    entry : "next_open", "close" or array-like of float
        The reference entry price convention. See step 1.
    tie : "sl_first" or "tp_first"
        What a side does when one bar touches both its target and its stop.
        "sl_first" (default) counts the stop; "tp_first" counts the target.
    events : array-like of int, optional
        Bar positions to label, strictly increasing, each <= len - 2.
        Default: every bar except the last (it has no bar after it).
    entry_offset : float or array-like of float
        How far fills land from the reference price, for example half the
        spread plus slippage (see `entry_offset()`). >= 0. Default 0.
    gap_exit : bool
        If True, a bar that opens past a barrier ends the side at that bar's
        open price, and that barrier counts as reached first (no tie).
        If False (default), exits are always at the barrier price.

    Returns
    -------
    BarrierResult
        Arrays with one value per event. Use ``.to_frame()`` for a table
        and ``.t1`` for label end bars (for purged cross-validation).

    Notes
    -----
    y = 0 happens in two ways. (a) No side reached its target: both were
    stopped (resolved) or at least one timed out (``unresolved`` is True).
    (b) Both sides reached their targets on the same bar. That can happen
    with tie="tp_first", and also with tp < sl under any tie rule: one wide
    bar can touch fill + tp and fill - tp while touching neither stop.

    Labels are targets for learning, not predictions or advice.
    """
    checked = check_inputs(
        open,
        high,
        low,
        close,
        tp=tp,
        sl=sl,
        horizon=horizon,
        entry=entry,
        tie=tie,
        entry_offset=entry_offset,
        gap_exit=gap_exit,
        events=events,
    )
    events_ = checked.events

    if checked.entry_mode == "next_open":
        ref_price = checked.open[events_ + 1]
    elif checked.entry_mode == "close":
        ref_price = checked.close[events_]
    else:
        ref_price = checked.entry_prices[events_]

    arrays = label_events(
        checked.open,
        checked.high,
        checked.low,
        checked.close,
        events_,
        np.ascontiguousarray(ref_price, dtype=np.float64),
        np.ascontiguousarray(checked.tp[events_]),
        np.ascontiguousarray(checked.sl[events_]),
        np.ascontiguousarray(checked.offset[events_]),
        checked.horizon,
        checked.tie == "tp_first",
        checked.gap_exit,
    )
    (
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
    ) = arrays

    return BarrierResult(
        index=events_,
        y=y,
        t1=t1,
        unresolved=unresolved,
        entry_price=np.asarray(ref_price, dtype=np.float64),
        up_excursion=up_excursion,
        down_excursion=down_excursion,
        long_outcome=long_outcome,
        long_t1=long_t1,
        long_exit_price=long_exit_price,
        long_ret_in_r=long_ret_in_r,
        short_outcome=short_outcome,
        short_t1=short_t1,
        short_exit_price=short_exit_price,
        short_ret_in_r=short_ret_in_r,
    )
