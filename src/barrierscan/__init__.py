"""barrierscan: first-touch (triple-barrier) labels for OHLC bars.

For each event bar, barrierscan places a target and a stop around an entry
price and scans the next bars. It records which barrier the bar highs and
lows touched first, or that time ran out. A slow, plain-Python oracle
(`brute_force_one`) applies the same rules so you can check any row.

Labels are targets for learning. They are not predictions or advice.

Method: Lopez de Prado (2018), Advances in Financial Machine Learning, ch. 3.
"""

from barrierscan.core import first_touch
from barrierscan.helpers import (
    average_true_range,
    breakeven_hit_rate,
    entry_offset,
    vol_barriers,
)
from barrierscan.oracle import brute_force_one
from barrierscan.result import BarrierResult

__version__ = "0.1.0"

__all__ = [
    "BarrierResult",
    "__version__",
    "average_true_range",
    "breakeven_hit_rate",
    "brute_force_one",
    "entry_offset",
    "first_touch",
    "vol_barriers",
]
