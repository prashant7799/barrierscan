# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-10-22

First release.

### Added

- `first_touch(open, high, low, close, *, tp, sl, horizon, entry, tie, events, entry_offset, gap_exit)`:
  a numba kernel that labels each event by the first barrier its next bars
  touch, using bar highs and lows. No column names are assumed.
- Per-bar `tp` and `sl` arrays (scalars are broadcast) and an optional
  `events` index of bar positions.
- Entry conventions: `"next_open"`, `"close"` or an array of custom entry
  prices.
- Same-bar tie rules: `"sl_first"` and `"tp_first"`.
- Gap-aware exits (`gap_exit=True`): a bar that opens past a barrier exits
  at its open.
- `entry_offset`: fills land this far from the reference price, against
  the trade.
- `BarrierResult` with `y`, `t1`, `unresolved`, `entry_price`,
  `up_excursion` and `down_excursion` (float64), per-side outcomes, end
  bars, exit prices and returns in R, plus `exit_price` and `ret_in_r` for
  the winning side, `to_frame()` (needs pandas) and `row()`.
- `brute_force_one`: a public, plain-Python oracle for one event.
- Helpers: `average_true_range`, `vol_barriers` (ATR barriers with a
  floor), `entry_offset` (half spread plus slippage) and
  `breakeven_hit_rate`.
- Documentation of every `y = 0` case, including the `tp < sl` case where
  one wide bar reaches both targets.
- A horizon longer than the data scans to the last bar. Horizons up to
  and past the 64-bit integer limit are capped safely.
- Tests: kernel against oracle on every row of many random series,
  hypothesis property tests, adversarial tests (invariance under dropped
  or cut bars, moved or scaled prices, extreme horizons, unusual input
  types), a close-only second oracle, hand cases, input checks, README
  output checks, and an installed-wheel check in CI.
- Examples: sensor thresholds, volatility barriers with a floor,
  meta-labeling, purged cross-validation with `t1`, and a synthetic
  benchmark method.

### Planned

- A release on PyPI, so `pip install barrierscan` works.
- A purged k-fold splitter that takes `t1` directly, with an embargo
  option.
- Timestamp support in `to_frame()`, mapping `index` and `t1` to times.
- Per-event horizons.
- A tie rule that flags same-bar ties instead of settling them.

[0.1.0]: https://github.com/prashant7799/barrierscan/releases/tag/v0.1.0
