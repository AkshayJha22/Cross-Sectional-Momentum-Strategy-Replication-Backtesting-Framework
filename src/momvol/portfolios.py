"""
Lee & Swaminathan (2000) portfolio machinery on a monthly stock panel.

At the end of each month t every eligible stock is ranked, independently,
  * into 10 price-momentum portfolios (R1 = losers .. R10 = winners) on its return over months t-J+1..t
  * into 3 volume portfolios (V1 = low .. V3 = high) on its average daily turnover over the same J months.
Portfolios are equal weighted. A K-month holding period is implemented the Jegadeesh-Titman way: the
return in month m is the equal-weighted average of the month-m returns of the K portfolios formed at
m-1, ..., m-K. That makes monthly observations non-overlapping, so simple t-statistics apply.

Differences from the paper (forced by the data, see the app for discussion):
  * monthly data, so the 1-week gap between formation and holding becomes either no gap (skip=0,
    default) or a full month (skip=1);
  * a stock needs J valid monthly returns, not two years of history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import nw_tstat, tstat


@dataclass
class Wide:
    ret: pd.DataFrame        # months x symbols, simple monthly return
    turnover: pd.DataFrame   # average daily turnover in the month (fraction)
    close: pd.DataFrame      # month-end raw price
    mktcap: pd.DataFrame


def to_wide(panel: pd.DataFrame) -> Wide:
    piv = lambda c: panel.pivot(index="month", columns="symbol", values=c).sort_index().astype("float64")
    return Wide(piv("ret"), piv("turnover"), piv("close"), piv("mktcap"))


def _rank_buckets(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Cross-sectional bucket 1..n per row (NaN where x is NaN)."""
    pct = x.rank(axis=1, pct=True, method="first")
    return np.ceil(pct * n).clip(1, n)


def form(w: Wide, J: int, skip: int = 0, min_price: float = 1.0, n_mom: int = 10, n_vol: int = 3):
    """Return (past_return, past_turnover, R labels, V labels) indexed by formation month."""
    lr = np.log1p(w.ret)
    past = np.expm1(lr.rolling(J, min_periods=J).sum()).shift(skip)
    turn = w.turnover.rolling(J, min_periods=max(1, J // 2)).mean().shift(skip)
    eligible = past.notna() & turn.notna() & (w.close >= min_price)
    past = past.where(eligible)
    turn = turn.where(eligible)
    return past, turn, _rank_buckets(past, n_mom), _rank_buckets(turn, n_vol)


def _forward_means(fwd: list[pd.DataFrame], mask: pd.DataFrame) -> list[pd.Series]:
    return [f.where(mask).mean(axis=1) for f in fwd]


def jt_returns(w: Wide, J: int, K: int, skip: int = 0, min_price: float = 1.0) -> pd.DataFrame:
    """Monthly returns (indexed by holding month) of every R x V cell plus R-only and V-only portfolios.

    Columns: 'R1V1'..'R10V3', 'R1'..'R10', 'V1'..'V3'.
    """
    _, _, R, V = form(w, J, skip, min_price)
    fwd = [w.ret.shift(-k) for k in range(1, K + 1)]            # row t holds the return of month t+k
    masks: dict[str, pd.DataFrame] = {}
    for r in range(1, 11):
        masks[f"R{r}"] = R == r
        for v in range(1, 4):
            masks[f"R{r}V{v}"] = (R == r) & (V == v)
    for v in range(1, 4):
        masks[f"V{v}"] = V == v
    out = {}
    for name, m in masks.items():
        per_vintage = _forward_means(fwd, m)
        # month-m return = average over k of the vintage formed at m-k
        aligned = pd.concat([s.shift(k) for k, s in enumerate(per_vintage, start=1)], axis=1)
        out[name] = aligned.mean(axis=1).where(aligned.notna().all(axis=1))
    res = pd.DataFrame(out)
    return res.dropna(how="all")


def event_time(w: Wide, J: int, years: int = 2, skip: int = 0, min_price: float = 1.0) -> dict[int, pd.DataFrame]:
    """Buy-and-hold returns of each cell in event years 1..years after formation (indexed by formation month).

    A stock that stops trading contributes the return it earned while listed (cash afterwards).
    """
    _, _, R, V = form(w, J, skip, min_price)
    lr = np.log1p(w.ret)
    out = {}
    for y in range(1, years + 1):
        yr = np.expm1(lr.rolling(12, min_periods=1).sum().shift(-12 * y))
        cols = {}
        for r in (1, 5, 10):
            cols[f"R{r}"] = yr.where(R == r).mean(axis=1)
            for v in range(1, 4):
                cols[f"R{r}V{v}"] = yr.where((R == r) & (V == v)).mean(axis=1)
        df = pd.DataFrame(cols)
        last_ok = w.ret.index[-1] - pd.DateOffset(months=12 * y)
        out[y] = df[df.index <= last_ok].dropna(how="all")
    return out


def add_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-cost spreads used in the paper."""
    d = df.copy()
    d["R10-R1"] = d["R10"] - d["R1"]
    for v in range(1, 4):
        d[f"R10-R1|V{v}"] = d[f"R10V{v}"] - d[f"R1V{v}"]
    for r in (1, 5, 10):
        if f"R{r}V3" in d:
            d[f"V3-V1|R{r}"] = d[f"R{r}V3"] - d[f"R{r}V1"]
    d["(R10-R1)V3 - (R10-R1)V1"] = d["R10-R1|V3"] - d["R10-R1|V1"]
    d["Early: R10V1-R1V3"] = d["R10V1"] - d["R1V3"]
    d["Late: R10V3-R1V1"] = d["R10V3"] - d["R1V1"]
    if "V1" in d:
        d["V1-V3"] = d["V1"] - d["V3"]
    return d


def table2(jt: pd.DataFrame) -> pd.DataFrame:
    """Paper Table II layout for one (J, K): mean monthly % and t-stats for R1/R5/R10/R10-R1 x V1/V2/V3/V3-V1."""
    d = add_spreads(jt)
    rows = []
    for r in ["R1", "R5", "R10", "R10-R1"]:
        mean_row, t_row = {"Portfolio": r, "stat": "mean %"}, {"Portfolio": r, "stat": "t"}
        for v in ["V1", "V2", "V3"]:
            s = d[f"{r}{v}"] if r != "R10-R1" else d[f"R10-R1|{v}"]
            mean_row[v], t_row[v] = 100 * s.mean(), tstat(s)
        s = d[f"V3-V1|{r}"] if r != "R10-R1" else d["(R10-R1)V3 - (R10-R1)V1"]
        mean_row["V3-V1"], t_row["V3-V1"] = 100 * s.mean(), tstat(s)
        rows += [mean_row, t_row]
    return pd.DataFrame(rows)


def formation_profile(w: Wide, J: int, skip: int = 0, min_price: float = 1.0) -> pd.DataFrame:
    """Paper Table I left block: formation-period return, turnover and price of R1, R5, R10."""
    past, turn, R, _ = form(w, J, skip, min_price)
    rows = []
    for r in (1, 5, 10):
        m = R == r
        geo = (1 + past.where(m)) ** (1 / J) - 1
        rows.append({"Portfolio": f"R{r}",
                     "formation return %/month": 100 * geo.mean(axis=1).mean(),
                     "avg daily turnover %": 100 * turn.where(m).mean(axis=1).mean(),
                     "median price $": w.close.where(m).median(axis=1).mean(),
                     "stocks per portfolio": m.sum(axis=1)[m.sum(axis=1) > 0].mean()})
    return pd.DataFrame(rows)


def grid_r10_r1(w: Wide, Js=(3, 6, 9, 12), Ks=(3, 6, 9, 12), skip: int = 0) -> pd.DataFrame:
    """Paper Table I right block: R10-R1 mean monthly return for every (J, K)."""
    rows = []
    for J in Js:
        for K in Ks:
            jt = add_spreads(jt_returns(w, J, K, skip))
            for name in ["R10-R1", "R10-R1|V1", "R10-R1|V3", "Early: R10V1-R1V3", "Late: R10V3-R1V1", "V1-V3"]:
                s = jt[name]
                rows.append({"J": J, "K": K, "series": name, "mean_pct": 100 * s.mean(), "t": tstat(s),
                             "months": int(s.notna().sum())})
    return pd.DataFrame(rows)


def event_table(ev: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Annual event-time returns (paper Tables VI/VII) with Newey-West (11 lag) t-stats."""
    rows = []
    for y, df in ev.items():
        d = df.copy()
        d["R10-R1"] = d["R10"] - d["R1"]
        d["Early: R10V1-R1V3"] = d["R10V1"] - d["R1V3"]
        d["Late: R10V3-R1V1"] = d["R10V3"] - d["R1V1"]
        d["V3-V1|R1"] = d["R1V3"] - d["R1V1"]
        d["V3-V1|R10"] = d["R10V3"] - d["R10V1"]
        for c in ["R10-R1", "Early: R10V1-R1V3", "Late: R10V3-R1V1", "V3-V1|R1", "V3-V1|R10"]:
            rows.append({"year": y, "series": c, "mean_pct": 100 * d[c].mean(), "nw_t": nw_tstat(d[c]),
                         "formations": int(d[c].notna().sum())})
    return pd.DataFrame(rows)
