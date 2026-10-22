"""The result object returned by `first_touch`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Column order used by `to_frame` and `row`. "bar" is the event's position.
COLUMNS = (
    "y",
    "t1",
    "unresolved",
    "entry_price",
    "up_excursion",
    "down_excursion",
    "exit_price",
    "ret_in_r",
    "long_outcome",
    "long_t1",
    "long_exit_price",
    "long_ret_in_r",
    "short_outcome",
    "short_t1",
    "short_exit_price",
    "short_ret_in_r",
)


@dataclass(frozen=True, eq=False)
class BarrierResult:
    """Labels for a set of events. Every field has one value per event.

    Combined label
        index           bar position of each event (int64)
        y               +1 long target first, -1 short target first, 0 no clear winner (int8)
        t1              bar where y was decided, or the last bar scanned (int64)
        unresolved      True when time ran out before y was decided (bool)
        entry_price     reference entry price, before any entry_offset (float64)
        up_excursion    highest high in bars index+1..t1, minus entry_price (float64)
        down_excursion  entry_price minus the lowest low in bars index+1..t1 (float64)

    Per side (each side is scanned on its own, to its own end)
        long_outcome / short_outcome         +1 target, -1 stop, 0 timeout (int8)
        long_t1 / short_t1                   bar where that side ended (int64)
        long_exit_price / short_exit_price   price the side exited at (float64)
        long_ret_in_r / short_ret_in_r       side's return in units of its stop distance (float64)

    `exit_price` and `ret_in_r` are properties: the winning side's values, NaN when y == 0.
    """

    index: np.ndarray
    y: np.ndarray
    t1: np.ndarray
    unresolved: np.ndarray
    entry_price: np.ndarray
    up_excursion: np.ndarray
    down_excursion: np.ndarray
    long_outcome: np.ndarray
    long_t1: np.ndarray
    long_exit_price: np.ndarray
    long_ret_in_r: np.ndarray
    short_outcome: np.ndarray
    short_t1: np.ndarray
    short_exit_price: np.ndarray
    short_ret_in_r: np.ndarray

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __repr__(self) -> str:
        n_up = int(np.sum(self.y == 1))
        n_down = int(np.sum(self.y == -1))
        n_zero = int(np.sum(self.y == 0))
        n_open = int(np.sum(self.unresolved))
        return (
            f"BarrierResult({len(self)} events: y=+1 {n_up}, y=-1 {n_down}, "
            f"y=0 {n_zero} of which unresolved {n_open})"
        )

    def _winning_side(self, long_values: np.ndarray, short_values: np.ndarray) -> np.ndarray:
        out = np.full(len(self), np.nan)
        long_won = self.y == 1
        short_won = self.y == -1
        out[long_won] = long_values[long_won]
        out[short_won] = short_values[short_won]
        return out

    @property
    def exit_price(self) -> np.ndarray:
        """Exit price of the side that won (y = +1: long, y = -1: short). NaN when y == 0."""
        return self._winning_side(self.long_exit_price, self.short_exit_price)

    @property
    def ret_in_r(self) -> np.ndarray:
        """Return in R of the side that won. NaN when y == 0."""
        return self._winning_side(self.long_ret_in_r, self.short_ret_in_r)

    def row(self, position: int) -> dict:
        """One event as a plain dict, in the same layout as `brute_force_one`.

        `position` counts events (0 .. len-1), not bars.
        """
        out: dict = {"bar": int(self.index[position])}
        for name in COLUMNS:
            out[name] = getattr(self, name)[position].item()
        return out

    def to_frame(self):
        """Return a pandas DataFrame indexed by event bar position. Needs pandas."""
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "to_frame needs pandas: pip install pandas (or barrierscan[pandas])"
            ) from error
        data = {name: getattr(self, name) for name in COLUMNS}
        return pd.DataFrame(data, index=pd.Index(self.index, name="bar"))
