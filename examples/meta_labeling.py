"""Per-side labels for meta-labeling, on a synthetic random walk.

A primary rule picks a side for each bar. Here the rule is a toy: go long
if the close rose over the last 10 bars, else go short. Meta-labeling asks
a second question: when the rule picked a side, did that side reach its
target before its stop? barrierscan scans each side on its own, so the
meta-label is one line of numpy.

A random walk has no edge, so the rule has none either. The end of the
script shows how much the labelling conventions alone move the hit rate.
Labels are targets for learning, not predictions or advice.
"""

import numpy as np
from synthetic import random_walk_ohlc

from barrierscan import breakeven_hit_rate, first_touch

open_, high, low, close = random_walk_ohlc(5000, seed=3)

LOOKBACK = 10
events = np.arange(LOOKBACK, len(close) - 1)  # bars with 10 bars of history
settings = {"tp": 2.0, "sl": 1.0, "horizon": 40, "events": events}
res = first_touch(open_, high, low, close, entry="next_open", tie="sl_first", **settings)

# The primary rule: +1 = long, -1 = short.
side = np.where(close[events] > close[events - LOOKBACK], 1, -1)


def meta_labels(result):
    """1 if the side the rule chose reached its target first, else 0."""
    return np.where(side == 1, result.long_outcome == 1, result.short_outcome == 1).astype(int)


meta_label = meta_labels(res)
print(" bar  side  long_outcome  short_outcome  meta_label")
for k in range(5):
    print(
        f"{res.index[k]:>4} {side[k]:>5} {res.long_outcome[k]:>13}"
        f" {res.short_outcome[k]:>14} {meta_label[k]:>11}"
    )

# The same events under other conventions. In this walk each bar opens at
# the previous close, so "next_open" and "close" give the same entry price.
tp_first = first_touch(open_, high, low, close, entry="next_open", tie="tp_first", **settings)
closes_only = first_touch(close, close, close, close, entry="close", **settings)

print()
print(f"events: {len(events)}")
print(f"break-even hit rate (tp = 2, sl = 1, no costs): {breakeven_hit_rate(2.0, 1.0):.3f}")
print("target hit rate of the chosen side:")
print(f"  OHLC bars, tie = sl_first   {meta_label.mean():.3f}")
print(f"  OHLC bars, tie = tp_first   {meta_labels(tp_first).mean():.3f}")
print(f"  closes only                 {meta_labels(closes_only).mean():.3f}")
