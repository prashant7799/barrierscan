"""Quickstart: label a tiny fridge-sensor series.

Each row is one minute of readings: the first, highest, lowest and last
temperature (deg C). For each minute we ask: within the next 4 minutes,
does the temperature rise 1.5 C first, fall 1.5 C first, or neither?
"""

import numpy as np

from barrierscan import first_touch

# Fridge sensor, one row per minute: first, highest, lowest, last reading (deg C)
first = np.array([4.0, 4.1, 4.6, 5.3, 6.2, 5.9, 4.8, 4.2])
high = np.array([4.2, 4.7, 5.4, 6.4, 6.5, 6.0, 4.9, 4.3])
low = np.array([3.9, 4.0, 4.5, 5.2, 5.8, 4.6, 4.1, 3.0])
last = np.array([4.1, 4.6, 5.3, 6.2, 5.9, 4.8, 4.2, 3.1])

res = first_touch(
    first,
    high,
    low,
    last,
    tp=1.5,  # up line: 1.5 C above the start
    sl=1.5,  # down line: 1.5 C below the start
    horizon=4,  # look at most 4 minutes ahead
    entry="close",  # start from this minute's last reading
    tie="sl_first",  # if one minute touches both lines, y = 0
)

# y: +1 = warmed by 1.5 C first, -1 = cooled by 1.5 C first, 0 = neither
print(res.to_frame()[["y", "t1", "unresolved", "up_excursion"]])
