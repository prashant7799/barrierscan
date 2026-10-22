"""Volatility-scaled barriers with a floor, on a synthetic random walk.

The walk has a quiet stretch and then a busy stretch. Barriers built from
the average true range (ATR) shrink in the quiet stretch, until the stop is
smaller than the bid-ask spread. A label built on such a stop describes a
move smaller than the cost of one trade. A floor tied to the spread keeps
the stop at a sensible minimum size.

Labels are targets for learning, not predictions or advice.
"""

import numpy as np
from synthetic import random_walk_ohlc

from barrierscan import first_touch, vol_barriers

# 1,000 quiet bars followed by 1,000 busy bars.
q_open, q_high, q_low, q_close = random_walk_ohlc(1000, seed=1, step=0.05, wick=0.05)
b_open, b_high, b_low, b_close = random_walk_ohlc(1000, seed=2, start=q_close[-1])
open_ = np.concatenate([q_open, b_open])
high = np.concatenate([q_high, b_high])
low = np.concatenate([q_low, b_low])
close = np.concatenate([q_close, b_close])

SPREAD = 0.2  # a synthetic bid-ask spread, the same on every bar

print(f"spread = {SPREAD}, floor = 3 x spread")
print("                  median sl   stops < spread   y=+1   y=-1    y=0")
for label, floor in (("no floor", 0.0), ("with floor", 3 * SPREAD)):
    tp, sl = vol_barriers(high, low, close, window=20, tp_mult=2.0, sl_mult=1.0, floor=floor)
    res = first_touch(open_, high, low, close, tp=tp, sl=sl, horizon=50, entry="next_open")
    for stretch, events in (("quiet", res.index < 1000), ("busy", res.index >= 1000)):
        stops = sl[res.index[events]]
        counts = [int(np.sum(res.y[events] == value)) for value in (1, -1, 0)]
        print(
            f"{label:<11} {stretch:<5} {np.median(stops):>9.2f}"
            f" {np.mean(stops < SPREAD):>16.0%}"
            f" {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}"
        )
