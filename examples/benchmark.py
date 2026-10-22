"""Synthetic benchmark: how long labelling takes on this machine.

This times the numba kernel against the plain-Python oracle on the same
synthetic random walk. It says nothing about other libraries. Numbers
depend on your machine, the horizon and how fast barriers are reached.

Method:
1. Build a random walk of N_BARS bars.
2. Call first_touch once on a tiny input, so numba compiles first and
   compile time is not counted.
3. Time first_touch on every bar (best of 3 runs).
4. Time brute_force_one on the first ORACLE_ROWS bars.
5. Print events per second for both.
"""

import time

import numpy as np
from synthetic import random_walk_ohlc

from barrierscan import brute_force_one, first_touch

N_BARS = 200_000
ORACLE_ROWS = 2_000
SETTINGS = {"tp": 3.0, "sl": 2.0, "horizon": 100, "entry": "next_open", "tie": "sl_first"}

open_, high, low, close = random_walk_ohlc(N_BARS, seed=0)

first_touch(open_[:50], high[:50], low[:50], close[:50], **SETTINGS)  # compile once

kernel_seconds = []
for _ in range(3):
    start = time.perf_counter()
    res = first_touch(open_, high, low, close, **SETTINGS)
    kernel_seconds.append(time.perf_counter() - start)
kernel_rate = len(res) / min(kernel_seconds)

# The oracle checks the whole input on every call, so give it a short slice.
part = slice(0, ORACLE_ROWS + SETTINGS["horizon"] + 1)
bars = (open_[part], high[part], low[part], close[part])
start = time.perf_counter()
for i in range(ORACLE_ROWS):
    brute_force_one(*bars, i, **SETTINGS)
oracle_rate = ORACLE_ROWS / (time.perf_counter() - start)

print(f"events labelled:       {len(res):,}")
print(f"median bars to label:  {np.median(res.t1 - res.index):.0f}")
print(f"first_touch:           {kernel_rate:,.0f} events per second")
print(f"brute_force_one:       {oracle_rate:,.0f} events per second")
