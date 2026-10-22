"""Tests for BarrierResult: fields, to_frame, row and repr."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
from synthetic import random_walk_ohlc

from barrierscan import BarrierResult, brute_force_one, first_touch
from barrierscan.result import COLUMNS

BARS = random_walk_ohlc(120, seed=11, gap=0.4)
KWARGS = {"tp": 2.0, "sl": 1.0, "horizon": 12, "entry": "close"}


def test_every_field_is_a_column():
    names = {field.name for field in fields(BarrierResult)} - {"index"}
    assert names <= set(COLUMNS)
    assert set(COLUMNS) - names == {"exit_price", "ret_in_r"}  # the two properties


def test_dtypes():
    res = first_touch(*BARS, **KWARGS)
    assert res.index.dtype == np.int64 and res.t1.dtype == np.int64
    assert res.y.dtype == np.int8 and res.long_outcome.dtype == np.int8
    assert res.unresolved.dtype == np.bool_
    assert res.up_excursion.dtype == np.float64 and res.down_excursion.dtype == np.float64


def test_to_frame():
    res = first_touch(*BARS, **KWARGS)
    frame = res.to_frame()
    assert list(frame.columns) == list(COLUMNS)
    assert frame.index.name == "bar" and len(frame) == len(res)
    assert np.array_equal(frame["y"].to_numpy(), res.y)
    assert np.array_equal(frame["ret_in_r"].to_numpy(), res.ret_in_r, equal_nan=True)


def test_combined_exit_follows_the_winner():
    res = first_touch(*BARS, **KWARGS)
    up, down, zero = res.y == 1, res.y == -1, res.y == 0
    assert np.array_equal(res.exit_price[up], res.long_exit_price[up])
    assert np.array_equal(res.exit_price[down], res.short_exit_price[down])
    assert np.all(np.isnan(res.exit_price[zero])) and np.all(np.isnan(res.ret_in_r[zero]))


def test_row_has_the_oracle_layout():
    res = first_touch(*BARS, **KWARGS)
    assert res.row(5).keys() == brute_force_one(*BARS, 5, **KWARGS).keys()
    assert isinstance(res.row(5)["y"], int) and isinstance(res.row(5)["unresolved"], bool)


def test_repr_counts_labels():
    res = first_touch(*BARS, **KWARGS)
    text = repr(res)
    assert text.startswith("BarrierResult(119 events")
    assert f"y=+1 {int(np.sum(res.y == 1))}" in text
