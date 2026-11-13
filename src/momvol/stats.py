"""Small, dependency-light statistics helpers used throughout the study."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def tstat(x: pd.Series) -> float:
    """Simple t-statistic of the mean (what Lee & Swaminathan report for monthly returns)."""
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def nw_tstat(x: pd.Series, lags: int = 11) -> float:
    """Newey-West t-statistic of the mean; used for overlapping (event-time) returns."""
    x = pd.Series(x).dropna()
    if len(x) < lags + 3:
        return np.nan
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def summary(x: pd.Series, periods_per_year: int = 12) -> dict:
    x = pd.Series(x).dropna()
    if x.empty:
        return {}
    wealth = (1 + x).cumprod()
    dd = wealth / wealth.cummax() - 1
    sd = x.std(ddof=1)
    return {
        "months": int(len(x)),
        "mean_pct": 100 * x.mean(),
        "t": tstat(x),
        "ann_mean_pct": 100 * x.mean() * periods_per_year,
        "ann_vol_pct": 100 * sd * np.sqrt(periods_per_year),
        "sharpe": x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
        "skew": float(x.skew()),
        "worst_month_pct": 100 * x.min(),
        "max_drawdown_pct": 100 * dd.min(),
        "hit_rate_pct": 100 * (x > 0).mean(),
    }


def factor_alpha(y: pd.Series, factors: pd.DataFrame, cols: list[str], excess: bool = False) -> dict:
    """Time-series regression y = a + b'F + e. y is a zero-cost spread unless excess=True."""
    df = pd.concat([y.rename("y"), factors], axis=1, join="inner").dropna()
    if len(df) < 24:
        return {}
    yy = df["y"] - (df["RF"] if excess else 0.0)
    X = sm.add_constant(df[cols])
    res = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out = {"alpha_pct": 100 * res.params["const"], "alpha_t": res.tvalues["const"], "r2": res.rsquared,
           "months": int(len(df))}
    for c in cols:
        out[f"b_{c}"] = res.params[c]
    return out
