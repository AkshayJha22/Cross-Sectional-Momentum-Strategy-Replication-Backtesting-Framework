"""Sanity tests for the portfolio machinery on synthetic data with known answers.

    python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from momvol import portfolios as P  # noqa: E402


def make_wide(n_stocks=300, n_months=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    # persistent per-stock drift => past winners keep winning (pure momentum world)
    drift = np.linspace(-0.02, 0.02, n_stocks)
    ret = pd.DataFrame(drift + rng.normal(0, 0.001, (n_months, n_stocks)), idx, cols)
    turnover = pd.DataFrame(np.tile(np.linspace(0.001, 0.02, n_stocks), (n_months, 1)), idx, cols)
    close = pd.DataFrame(20.0, idx, cols)
    return P.Wide(ret, turnover, close, close * 1e6)


def test_buckets_are_balanced():
    w = make_wide()
    _, _, R, V = P.form(w, J=6)
    last = R.iloc[-1].dropna()
    assert sorted(last.unique()) == list(range(1, 11))
    assert last.value_counts().nunique() == 1                      # 30 stocks in every decile
    assert sorted(V.iloc[-1].dropna().unique()) == [1, 2, 3]


def test_momentum_spread_matches_drift():
    w = make_wide()
    jt = P.add_spreads(P.jt_returns(w, J=6, K=3))
    # top decile drift averages ~+0.018, bottom ~-0.018 => spread ~ 0.036 per month
    assert abs(jt["R10-R1"].mean() - 0.036) < 0.002


def test_jt_requires_all_vintages():
    w = make_wide(n_months=20)
    jt = P.jt_returns(w, J=6, K=6)
    # first formation at month 6 (index 5); full K=6 vintages first available 6 months later
    assert jt["R1"].first_valid_index() == w.ret.index[5 + 6]


def test_no_lookahead():
    """Changing returns after formation must not change formation labels."""
    w = make_wide()
    _, _, R1, _ = P.form(w, J=6)
    w2 = make_wide()
    w2.ret.iloc[20:] = 0.5
    _, _, R2, _ = P.form(w2, J=6)
    pd.testing.assert_frame_equal(R1.iloc[:20], R2.iloc[:20])
