"""`brute_force_one`: a slow, plain-Python reference for one event.

This is the oracle. It applies the rules of `first_touch` one bar at a
time, with no numba and no shared labelling code. The tests compare the
fast kernel with it on many random series. You can call it yourself to
check any row of your own data.
"""

from __future__ import annotations

import math

from barrierscan._validate import check_inputs


def brute_force_one(
    open,
    high,
    low,
    close,
    i: int,
    *,
    tp,
    sl,
    horizon: int,
    entry="next_open",
    tie: str = "sl_first",
    entry_offset=0.0,
    gap_exit: bool = False,
) -> dict:
    """Label the single event at bar ``i`` the slow, obvious way.

    Takes the same arguments as `first_touch` (with one bar ``i`` instead of
    ``events``) and returns a dict with the same keys as
    ``BarrierResult.row``: "bar", "y", "t1", "unresolved", "entry_price",
    "up_excursion", "down_excursion", "exit_price", "ret_in_r", and the
    long_* and short_* fields.
    """
    if isinstance(i, bool) or not hasattr(i, "__index__"):
        raise TypeError("i must be an integer bar position")
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
        events=[int(i)],
    )
    i = int(i)
    opens = checked.open.tolist()
    highs = checked.high.tolist()
    lows = checked.low.tolist()
    closes = checked.close.tolist()
    n_bars = len(closes)

    if checked.entry_mode == "next_open":
        reference = opens[i + 1]
    elif checked.entry_mode == "close":
        reference = closes[i]
    else:
        reference = float(checked.entry_prices[i])

    tp_i = float(checked.tp[i])
    sl_i = float(checked.sl[i])
    offset_i = float(checked.offset[i])
    last_bar = min(i + checked.horizon, n_bars - 1)
    window = list(range(i + 1, last_bar + 1))

    def run_side(direction: int) -> tuple[int, int, float, float]:
        """direction +1 is the long side, -1 the short side.

        Returns (outcome, end bar, exit price, fill price).
        """
        fill = reference + direction * offset_i
        target = fill + direction * tp_i
        stop = fill - direction * sl_i
        for j in window:
            # For this side, "best" is the most favourable price in the bar
            # and "worst" the least favourable.
            best = highs[j] if direction == 1 else lows[j]
            worst = lows[j] if direction == 1 else highs[j]
            # direction * (price - level) >= 0 means: price is at the level
            # or beyond it in the profit direction.
            if checked.gap_exit and direction * (opens[j] - target) >= 0:
                return 1, j, opens[j], fill
            if checked.gap_exit and direction * (opens[j] - stop) <= 0:
                return -1, j, opens[j], fill
            reached_target = direction * (best - target) >= 0
            reached_stop = direction * (worst - stop) <= 0
            if reached_target and reached_stop:
                if checked.tie == "tp_first":
                    return 1, j, target, fill
                return -1, j, stop, fill
            if reached_target:
                return 1, j, target, fill
            if reached_stop:
                return -1, j, stop, fill
        return 0, last_bar, closes[last_bar], fill

    sides = {"long": run_side(+1), "short": run_side(-1)}

    # Which sides reached their target, and on which bar?
    winners = {name: result[1] for name, result in sides.items() if result[0] == 1}
    if winners:
        first_win_bar = min(winners.values())
        first_winners = [name for name, bar in winners.items() if bar == first_win_bar]
        if len(first_winners) == 2:
            y = 0  # both targets on the same bar
        elif first_winners == ["long"]:
            y = 1
        else:
            y = -1
        t1 = first_win_bar
        unresolved = False
    else:
        y = 0
        t1 = max(sides["long"][1], sides["short"][1])
        unresolved = sides["long"][0] == 0 or sides["short"][0] == 0

    up_excursion = max(highs[i + 1 : t1 + 1]) - reference
    down_excursion = reference - min(lows[i + 1 : t1 + 1])

    def side_fields(name: str, direction: int) -> dict:
        outcome, end_bar, exit_price, fill = sides[name]
        return {
            f"{name}_outcome": outcome,
            f"{name}_t1": end_bar,
            f"{name}_exit_price": exit_price,
            f"{name}_ret_in_r": direction * (exit_price - fill) / sl_i,
        }

    long_fields = side_fields("long", +1)
    short_fields = side_fields("short", -1)
    if y == 1:
        exit_price = long_fields["long_exit_price"]
        ret_in_r = long_fields["long_ret_in_r"]
    elif y == -1:
        exit_price = short_fields["short_exit_price"]
        ret_in_r = short_fields["short_ret_in_r"]
    else:
        exit_price = math.nan
        ret_in_r = math.nan

    return {
        "bar": i,
        "y": y,
        "t1": t1,
        "unresolved": unresolved,
        "entry_price": reference,
        "up_excursion": up_excursion,
        "down_excursion": down_excursion,
        "exit_price": exit_price,
        "ret_in_r": ret_in_r,
        **long_fields,
        **short_fields,
    }
