# barrierscan

barrierscan labels rows of bar data by asking one question per row: within
the next N bars, did the value go up by X first, down by Y first, or
neither? It reads each bar's open, high, low and close (OHLC), so it sees a
touch in the middle of a bar, not just at the close. Bars cannot tell you
some things, so each one is an explicit setting: where the entry is, what
happens when one bar touches both lines, and what happens on a gap. It
also ships a slow, plain-Python oracle that applies the same rules, so you
can check any row yourself. It is a version of the triple-barrier method
from López de Prado (2018), chapter 3. It is for data scientists,
students, teachers and Kaggle competitors who need these labels for
machine learning, for example for triple-barrier targets or meta-labeling,
on price bars or on any other bar data such as sensor readings.

## What it does NOT do

- It is not a backtester. It does not replay a strategy, size positions or
  track a portfolio.
- It knows one cost only: an optional entry offset (for example half the
  spread plus slippage). There are no fees, exit costs or funding.
- Labels describe what already happened in past data. They are targets for
  learning, not predictions, signals or advice.
- It makes no speed claim against other libraries. Its value is clear
  written rules and a checker you can call.
- It does not fetch data, and it does not use timestamps. It works on bar
  positions.
- It does not split your data for you. It gives you `t1` so that your
  splitter can purge overlapping rows.
- It does not pick events for you (there is no CUSUM filter), and it does
  not compute sample weights or label uniqueness (chapter 4 of the book).
- It does not label a timeout by the sign of its return, as some versions
  of the method do. A timeout gives `y = 0`. The per-side returns are there
  if you want a sign.
- It does not turn returns into barriers for you. `tp` and `sl` are price
  distances. For a barrier of about 1%, pass `tp=0.01 * close`.

## Install

```bash
pip install git+https://github.com/prashant7799/barrierscan
```

It needs Python 3.10 or newer, numpy and numba. `to_frame()` also needs
pandas: `pip install "barrierscan[pandas] @ git+https://github.com/prashant7799/barrierscan"`.
A PyPI package is planned.

## Quickstart (60 seconds)

A fridge sensor gives one row per minute: the first, highest, lowest and
last reading. For each minute, did the temperature rise 1.5 °C first, fall
1.5 °C first, or neither, within the next 4 minutes?

```python
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
```

Output:

```text
     y  t1  unresolved  up_excursion
bar                                 
0    1   3       False           2.3
1    1   3       False           1.8
2    0   6        True           1.2
3   -1   5       False           0.3
4   -1   6       False           0.1
5   -1   7       False           0.1
6    0   7        True           0.1
```

Row 0 starts at 4.1 °C. Minute 3 reaches 6.4, past the up line at 5.6, so
y = +1 and t1 = 3. Row 2 touches neither line in 4 minutes, so it is
unresolved. Row 6 runs out of data. `up_excursion` is the biggest rise seen
while waiting. The last minute is not a row: it has no minute after it.

The same script is in [`examples/quickstart.py`](examples/quickstart.py).

## Concepts

### One event, one side

For each event bar `i`, barrierscan picks a reference entry price, places a
target and a stop, and scans the next bars. This picture shows the long
side. Here the target is touched first, so the long side wins.

```text
 price
   |                                                first touch
   |                                                     |
   |   target  - - - - - - - - - - - - - - - - - - - - - x - -   fill + tp
   |                                          /\        /
   |                                  /\     /  \  /\  /
   |   entry   o- - - - - - \    /\  /  \   /    \/  \/
   |           |             \  /  \/    \_/
   |           |              \/
   |   stop    - - - - - - - - - - - - - - - - - - - - - - - -   fill - sl
   |           |
   |           |<------ horizon: bars i+1 ... i+horizon ------>|
   +-----------+-----------------------------------------------+---> bars
             bar i
```

The short side is the mirror image. Both sides are scanned for every event.

| | long side | short side |
|---|---|---|
| fill price | reference + entry_offset | reference − entry_offset |
| target | fill + tp | fill − tp |
| stop | fill − sl | fill + sl |
| target touched when | high ≥ target | low ≤ target |
| stop touched when | low ≤ stop | high ≥ stop |

"Touched" includes reaching the price exactly.

### Entry convention

- `entry="next_open"` (default): the reference price is `open[i + 1]`. You
  decide at the end of bar `i` and fill at the next open. Bar `i + 1` is
  scanned in full, because all of its prices come at or after its open.
- `entry="close"`: the reference price is `close[i]`.
- `entry=array`: your own price for each bar, for example a limit price.
  Pick prices you could really have had at the end of bar `i`. Otherwise
  the labels look into the future.

In all three cases the scan covers bars `i + 1` to `min(i + horizon, last bar)`.

### Same-bar tie rule

A bar shows its high and low, but not which came first. If one bar touches
both the target and the stop of a side, the `tie` rule decides:

- `tie="sl_first"` (default): the stop counts. This is the cautious choice.
- `tie="tp_first"`: the target counts.

### Gap-aware exits

With `gap_exit=True`, a bar that opens past a barrier ends the side at that
bar's open price. A stop then fills at or past the stop price, against
you. A target fills at or past the target price, in your favour. The open
also settles the order: that barrier was clearly reached first, so no tie
rule is needed. With `gap_exit=False` (default), every exit is at the
barrier price.

### The combined label y

- `y = +1`: the long target was touched first.
- `y = -1`: the short target was touched first.
- `y = 0` in every other case:
  1. Both sides were stopped before either target. The label is resolved.
  2. At least one side timed out and neither won. `unresolved` is True.
     This also covers running out of data.
  3. Both targets were touched on the same bar. The label is resolved.
     This happens with `tie="tp_first"` on a wide bar, and also with
     `tp < sl` under either tie rule (see below).

### When tp < sl

With `tp < sl`, the two targets (fill + tp and fill − tp) sit inside the two
stops. One wide bar can touch both targets and neither stop. Then both
sides win on the same bar, both per-side outcomes are +1, and `y = 0`. The
tie rule does not change this, because no side touched its own stop.

### t1, per-side outcomes and returns

- `t1` is the bar where `y` was decided: the winning bar, the bar where the
  second side was stopped, or the last bar scanned. Use it to purge
  overlapping rows in cross-validation (López de Prado 2018, chapter 7).
- Each side is scanned on its own until it ends, even after `y` is decided.
  `long_outcome` and `short_outcome` are +1 (target), −1 (stop) or 0
  (timeout). `long_t1` and `short_t1` are where each side ended. Use these
  for meta-labeling, where a primary model picks the side.
- `long_ret_in_r` is `(exit − fill) / sl` and `short_ret_in_r` is
  `(fill − exit) / sl`. Without gaps a win is `tp / sl` and a loss is −1.
  A timeout exits at the close of the last bar scanned.
- `up_excursion` and `down_excursion` are the biggest move up and down from
  the reference price over bars `i + 1` to `t1`. They can be negative.

## API reference

```python
from barrierscan import (
    first_touch,
    BarrierResult,
    brute_force_one,
    vol_barriers,
    average_true_range,
    entry_offset,
    breakeven_hit_rate,
)
```

### first_touch

```text
first_touch(open, high, low, close, *, tp, sl, horizon, entry="next_open",
            tie="sl_first", events=None, entry_offset=0.0, gap_exit=False)
    -> BarrierResult
```

| argument | meaning |
|---|---|
| `open, high, low, close` | 1-D arrays, one value per bar, same length, finite. `low ≤ open, close ≤ high`. Names do not matter. |
| `tp, sl` | Target and stop distances in price units. A scalar, or one value per bar (bar `i`'s value is used for the event at bar `i`). Must be finite and > 0 at every event. |
| `horizon` | Most bars to scan after the event bar. An int ≥ 1. A horizon longer than the data scans to the last bar. |
| `entry` | `"next_open"`, `"close"`, or an array of entry prices (one per bar). |
| `tie` | `"sl_first"` or `"tp_first"`. |
| `events` | Bar positions to label: integers, strictly increasing, at most `len − 2`. Default: every bar but the last. For a True/False mask, pass `np.flatnonzero(mask)`. |
| `entry_offset` | How far fills land from the reference price. A scalar or one value per bar, ≥ 0. |
| `gap_exit` | `True` to exit at the open when a bar opens past a barrier. |

Bad inputs raise `ValueError` or `TypeError` with a clear message. Where
there is a bad bar, the message names the first one.

### BarrierResult

A frozen dataclass of numpy arrays, one value per event.

| field | dtype | meaning |
|---|---|---|
| `index` | int64 | bar position of each event |
| `y` | int8 | +1, −1 or 0 (see Concepts) |
| `t1` | int64 | bar where `y` was decided, or the last bar scanned |
| `unresolved` | bool | True when time ran out before `y` was decided |
| `entry_price` | float64 | reference entry price (before `entry_offset`) |
| `up_excursion`, `down_excursion` | float64 | biggest move up / down over bars `i + 1 .. t1` |
| `long_outcome`, `short_outcome` | int8 | +1 target, −1 stop, 0 timeout |
| `long_t1`, `short_t1` | int64 | bar where each side ended |
| `long_exit_price`, `short_exit_price` | float64 | price each side exited at |
| `long_ret_in_r`, `short_ret_in_r` | float64 | each side's return in units of `sl` |

Also:

- `exit_price`, `ret_in_r`: properties with the winning side's values, NaN
  when `y = 0`.
- `to_frame()`: a pandas DataFrame indexed by event bar (`"bar"`). Needs pandas.
- `row(k)`: event number `k` as a dict, in the same layout as `brute_force_one`.
- `len(result)`: the number of events.

### brute_force_one

```text
brute_force_one(open, high, low, close, i, *, tp, sl, horizon, entry="next_open",
                tie="sl_first", entry_offset=0.0, gap_exit=False) -> dict
```

The oracle. It labels the single event at bar `i` in plain Python, one bar
at a time, and returns a dict with the keys of `BarrierResult.row`. It is
slow on purpose. Use it to check rows of your own data:

```python
import math

settings = {"tp": 2.0, "sl": 1.0, "horizon": 30}
res = first_touch(open_, high, low, close, **settings)

k = 100  # any event number
fast = res.row(k)
slow = brute_force_one(open_, high, low, close, int(res.index[k]), **settings)
for key, value in slow.items():
    # exit_price and ret_in_r are NaN when y = 0, and NaN != NaN
    assert fast[key] == value or (math.isnan(fast[key]) and math.isnan(value)), key
```

### Helpers

- `average_true_range(high, low, close, window=14)`: trailing mean of the
  true range. Bar `i` uses bars `i − window + 1 .. i` only, so it is safe
  for an event at bar `i`. The first bars average what exists so far.
- `vol_barriers(high, low, close, *, window=14, tp_mult=2.0, sl_mult=1.0, floor=0.0) -> (tp, sl)`:
  `sl = max(sl_mult × ATR, floor)` and `tp = sl × tp_mult / sl_mult`. The
  floor (a scalar or one value per bar) stops barriers from shrinking below,
  say, a few spreads in quiet periods.
- `entry_offset(spread, slippage=0.0)`: `spread / 2 + slippage`, ready for
  `first_touch(entry_offset=...)`.
- `breakeven_hit_rate(tp, sl, cost=0.0)`: `(sl + cost) / (tp + sl)`, the
  target hit rate at which a trade that gains `tp` or loses `sl` breaks
  even. It ignores timeouts.

## Examples

The quickstart uses eight made-up sensor readings. The other examples use
synthetic random walks. Run them from the repository root, for example
`python examples/purged_cv.py`. The outputs below were produced by the
scripts as they are, and a test checks that they still match.

### Sensor thresholds within a time cap

[`examples/quickstart.py`](examples/quickstart.py): the fridge sensor from
the quickstart. Nothing about it is financial.

### Volatility-scaled barriers with a floor

[`examples/vol_barriers_floor.py`](examples/vol_barriers_floor.py) builds a
walk with a quiet stretch and a busy stretch, then labels it with ATR-based
barriers, once without a floor and once with a floor of 3 × spread.

```text
spread = 0.2, floor = 3 x spread
                  median sl   stops < spread   y=+1   y=-1    y=0
no floor    quiet      0.12             100%    216    334    450
no floor    busy       2.41               0%    243    279    477
with floor  quiet      0.60               0%      0     37    963
with floor  busy       2.41               0%    243    280    476
```

Without a floor, every quiet-stretch stop is smaller than the spread, yet
550 rows get a non-zero label. Those moves are too small to matter. With the
floor, the quiet stretch is almost all `y = 0`: nothing moved enough to
matter. Its 37 non-zero rows are all in its last 50 bars, whose 50-bar
windows reach into the busy stretch. The busy stretch barely changes,
because its stops are already above the floor.

### Per-side labels for meta-labeling

[`examples/meta_labeling.py`](examples/meta_labeling.py) uses a toy primary
rule (follow the last 10-bar move) to pick a side. The meta-label is 1 when
the chosen side reached its target first:

```python
meta_label = np.where(side == 1, res.long_outcome == 1, res.short_outcome == 1).astype(int)
```

```text
 bar  side  long_outcome  short_outcome  meta_label
  10    -1            -1              1           1
  11    -1            -1              1           1
  12    -1            -1              1           1
  13    -1            -1              1           1
  14    -1            -1             -1           0

events: 4989
break-even hit rate (tp = 2, sl = 1, no costs): 0.333
target hit rate of the chosen side:
  OHLC bars, tie = sl_first   0.233
  OHLC bars, tie = tp_first   0.327
  closes only                 0.363
```

A random walk has no edge, so the rule has none. Yet the hit rate moves
from 0.233 to 0.363 with the labelling convention alone. Close-only labels
miss the intrabar touches of the nearer stop, so here they look better than
break-even. This is why barrierscan makes the conventions explicit.

### t1 for purged cross-validation

[`examples/purged_cv.py`](examples/purged_cv.py) shows the leak first, then
the fix. The "model" only copies the label of the training row closest in
time. On a random walk it should score 0.5.

```text
20 random walks x 10 folds; labels look up to 60 bars ahead
                     test rows sharing future   mean test
                     bars with a training row   accuracy
shuffled folds                          100.0%      0.874
purged block folds                        0.0%      0.497
```

Shuffled folds score 0.874 because every test row has a training neighbour
whose label window overlaps its own. Contiguous folds that purge every
training row whose `[bar, t1]` overlaps the test block score 0.497, a coin
flip, as they should.

### Synthetic benchmark method

[`examples/benchmark.py`](examples/benchmark.py) times `first_touch` and
`brute_force_one` on a 200,000-bar random walk. It compiles once before
timing, takes the best of 3 runs, and times the oracle on 2,000 rows. It
prints events per second for both. The numbers depend on your machine, the
horizon and the barriers, so none are quoted here, and it says nothing
about other libraries.

## How correctness is tested

Run `pytest -q`. The suite has 250 tests and uses synthetic data only.
None of them use the network.

- **Kernel against oracle, every row.** 36 configurations (3 entry modes ×
  2 tie rules × gaps on/off × tp > sl, tp = sl, tp < sl), 4 random walks
  each, with per-bar tp, sl, offsets and custom entries. Every field of
  every row must be exactly equal: about 36,000 rows. Half of the walks use
  prices on a 0.25 grid, so exact touches and same-bar ties are common. A
  separate test checks that these runs really hit ties, gaps, timeouts,
  both-stopped rows and both-targets-on-one-bar rows.
- **Property tests** (hypothesis, 200 cases each): kernel equals oracle;
  `t1` stays inside the scan window; winners travelled at least `tp`;
  `unresolved` means no winner and a timeout; flipping prices upside down
  swaps the long and short results exactly; a longer horizon never changes
  a resolved result; labelling a subset of events gives the same rows.
- **Adversarial tests** (hypothesis, 200 cases each, plus fixed cases).
  Changes that must not change the labels really do not: dropping earlier
  bars, cutting bars after the scan window, moving every price, and scaling
  prices and distances together. `y` and `t1` must follow from the two
  sides. The tie rule and gap exits never move the bar where a side ends,
  and gap exits are never better than the barrier. Also: huge horizons (up
  to the int64 limit and past it), a two-bar input, lists, float32,
  big-endian, read-only and pandas inputs, and a check that inputs are
  never changed.
- **A second, close-only oracle**, written with numpy matrices instead of a
  loop. On flat bars (open = high = low = close) both must agree exactly. On
  real OHLC bars, the labeller must never resolve a side later than the
  close-only oracle does.
- **Hand cases** for each rule: tie rules, entry conventions, gaps, entry
  offsets, the tp < sl case, timeouts and end of data.
- **Input checks**: bad lengths, NaN, inconsistent bars, bad barriers,
  horizons, events and options all raise clear errors.
- **This README**: tests run the quickstart code, the oracle snippet and
  the example scripts, and compare their output with the blocks shown here.
- **The installed wheel**: CI builds a wheel, installs it in a clean
  environment with `NUMBA_CACHE_DIR` set, and runs the quickstart.

## Limitations and assumptions

- Bars hide the order of prices inside a bar. The tie rule is a
  convention, not a fact. `sl_first` is the cautious choice.
- `gap_exit` assumes the open is the first price of the bar.
- With `gap_exit=False`, the open is not used to order touches. A bar that
  opens past the target and then falls to the stop is a same-bar tie, so
  `sl_first` counts the stop. Use `gap_exit=True` if your opens are real
  first prices.
- If `entry_offset` is at least `sl`, the stop sits at or beyond the
  reference price. With `entry="next_open"` that side is stopped on the
  first bar.
- Barriers are fixed when the event starts. There are no trailing stops.
- One `horizon` applies to every event.
- Bars are treated as consecutive. If your data has missing bars, a
  horizon of N bars is not N units of time.
- `t1` is a bar position, not a timestamp. Map it with `times[res.t1]`.
- Inputs must be finite. Drop or fill missing bars first.
- Excursions are measured from the reference price, not the fill, and use
  the full range of the bar at `t1`.
- The first call compiles the kernel with numba, which takes a second or
  two. The compiled code is cached on disk. Numba tends to support a new
  Python version a few months after its release.
- Work grows with events × bars scanned. Far barriers with a long horizon
  are the slowest case. The kernel runs on one CPU core.

## Related tools

Several open-source libraries also build triple-barrier or similar labels.
Most of them work on close prices with timestamps and set the barriers as
returns. Many also include event filters, sample weights and many more
methods from the book. Some work on trade-by-trade (tick) data, where every
price is seen, so the order of prices inside a bar is not a problem there.

Choose one of those if you want a full toolkit, timestamps or tick data.
Choose barrierscan if you label OHLC bars and want intrabar high/low
touches with a written tie rule and entry convention, gap exits, per-side
outcomes, `t1`, and an oracle you can call, in a small MIT package with
two direct dependencies.

## References and citation

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. ISBN 978-1-119-48208-6. Chapter 3 (labeling: the triple-barrier
  method and meta-labeling) and chapter 7 (cross-validation: purging).

If barrierscan helps your work, please cite the book, and optionally this
package:

```bibtex
@software{lohar2026barrierscan,
  author  = {Lohar, Prashant},
  title   = {barrierscan: first-touch labels for OHLC bars},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/prashant7799/barrierscan}
}
```

## Licence

MIT. Copyright (c) 2026 Prashant Lohar. See [LICENSE](LICENSE).
