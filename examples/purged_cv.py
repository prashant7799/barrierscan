"""Use t1 to purge training rows, on synthetic random walks.

Each label looks ahead up to `horizon` bars, so the labels of nearby bars
share most of their future. A shuffled train/test split puts those
neighbours on both sides of the split, and a model can "predict" a test
label by copying its neighbour's label. That is leakage, not skill.

The prices are pure random walks, so no model can beat a coin flip on
new data. A test accuracy clearly above 50% is leakage.

The fix (Lopez de Prado 2018, ch. 7): split into contiguous blocks and drop
("purge") every training row whose label window [bar, t1] overlaps the
label windows of the test block.
"""

import numpy as np
from synthetic import random_walk_ohlc

from barrierscan import first_touch

N_WALKS = 20
N_FOLDS = 10


def windows_overlap(start_a, end_a, start_b, end_b):
    """True where window [start_a, end_a] overlaps window [start_b, end_b]."""
    return (start_a <= end_b) & (end_a >= start_b)


def evaluate(bars, labels, t1, train, test):
    """Return (share of test rows sharing future bars with training, accuracy).

    The "model" copies the label of the training row closest in time. It
    uses no prices at all. Any feature that changes slowly over time (a
    moving average, a regime flag, a calendar) lets a model do the same.
    """
    shares = windows_overlap(
        bars[test][:, None], t1[test][:, None], bars[train][None, :], t1[train][None, :]
    )
    distance = np.abs(bars[test][:, None] - bars[train][None, :])
    nearest = train[distance.argmin(axis=1)]
    accuracy = np.mean(labels[nearest] == labels[test])
    return np.mean(shares.any(axis=1)), accuracy


results = {"shuffled folds": [], "purged block folds": []}
for seed in range(N_WALKS):
    open_, high, low, close = random_walk_ohlc(3000, seed=seed)
    res = first_touch(open_, high, low, close, tp=4.0, sl=4.0, horizon=60, entry="close")

    # Keep the rows with a winner, so the task is a plain up-or-down guess.
    keep = res.y != 0
    bars, labels, t1 = res.index[keep], res.y[keep], res.t1[keep]
    rows = np.arange(len(bars))

    # Wrong: shuffled folds.
    rng = np.random.default_rng(seed)
    for test in np.array_split(rng.permutation(rows), N_FOLDS):
        train = np.setdiff1d(rows, test)
        results["shuffled folds"].append(evaluate(bars, labels, t1, train, test))

    # Right: contiguous folds. Purge every training row whose window
    # [bar, t1] overlaps [first test bar, last bar any test label looks at].
    for test in np.array_split(rows, N_FOLDS):
        leaks = windows_overlap(bars, t1, bars[test].min(), t1[test].max())
        train = rows[~leaks]
        results["purged block folds"].append(evaluate(bars, labels, t1, train, test))

print(f"{N_WALKS} random walks x {N_FOLDS} folds; labels look up to 60 bars ahead")
print("                     test rows sharing future   mean test")
print("                     bars with a training row   accuracy")
for name, values in results.items():
    share, accuracy = np.mean(values, axis=0)
    print(f"{name:<20} {share:>25.1%} {accuracy:>10.3f}")
