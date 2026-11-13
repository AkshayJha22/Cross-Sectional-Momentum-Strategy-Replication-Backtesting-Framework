"""
Build the monthly stock panel used for the stock-level replication.

Source: eliangcs/pystock-data (https://github.com/eliangcs/pystock-data, CC-BY-SA-4.0)
  * 0001-0003_initial.tar.gz : daily prices 2009-01-02 .. 2015-03-20 for symbols listed in Mar 2015
                               (close = raw, adj_close = split+dividend adjusted, volume = split adjusted)
                               plus 10-K / 10-Q summary data scraped from SEC EDGAR
  * YYYYMMDD.tar.gz          : one file per trading day 2015-03-23 .. 2017-03-31, captured live
                               (close = volume = raw, as of that day; includes stocks that later delisted)

Everything is converted to one "basis" (share counts as of 2015-03-20) so that
turnover = shares traded / shares outstanding is consistent across stock splits:

  pre-2015  : a_t = close_t / adj_close_t  (cumulative split * dividend factor)
              shares_basis = raw_shares_at_report * a_report
              volume is already in basis units
  post-2015 : splits are detected from raw close-to-close moves that match a standard split
              ratio (2:1, 3:2, 1:10, ...) and accumulated into C_t, so
              price_basis = close * C_t, volume_basis = volume / C_t, shares_basis = raw_shares / C_report

Shares outstanding are not in the source, so they are backed out of each filing as
net_income / eps_basic (weighted-average shares for the period). They are used only to rank
stocks into turnover terciles, where a few percent of error does not change the ranking.

Usage:
    python scripts/build_panel.py /path/to/extracted/pystock-data  data/panel_monthly.parquet
where the input folder holds one sub-folder per archive (0001_initial/, 20150323/, ...),
each containing prices.csv and reports.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_N = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50], dtype=float)
# price multiplier on the split day: N:1 forward split -> 1/N, 1:N reverse split -> N, 3:2 -> 2/3
# Odd ratios (3:2, 4:3, 5:4) are left out on purpose: in this data they matched far more earnings
# crashes than real splits, and wrongly "repairing" a crash would bias the loser portfolio upwards.
SPLIT_RATIOS = np.concatenate([1 / _N, _N])
REPORT_COLS = ["symbol", "end_date", "amend", "period_focus", "fiscal_year", "doc_type", "revenues",
               "op_income", "net_income", "eps_basic", "eps_diluted", "dividend", "assets", "cur_assets",
               "cur_liab", "cash", "equity", "cash_flow_op", "cash_flow_inv", "cash_flow_fin"]
REPORT_LAG_DAYS = 60          # filings become usable 60 days after the period end
MIN_PRICE_ROWS = 40           # ignore symbols with almost no history


def _clean_symbol_mask(s: pd.Series) -> pd.Series:
    # keep plain tickers; drops preferreds, units, warrants, rights (e.g. ABC-PA, ABC.U, ABCW)
    return s.str.fullmatch(r"[A-Z]{1,5}")


def load_initial(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(root.glob("000*_initial")):
        df = pd.read_csv(d / "prices.csv", usecols=["symbol", "date", "close", "volume", "adj_close"],
                         dtype={"symbol": str})
        frames.append(df)
    px = pd.concat(frames, ignore_index=True)
    px = px[_clean_symbol_mask(px["symbol"])]
    for c in ["close", "volume", "adj_close"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    px = px[(px["close"] > 0) & (px["adj_close"] > 0)]
    px["a"] = px["close"] / px["adj_close"]                      # >= 1 before splits/dividends
    px["ret"] = px.groupby("symbol")["adj_close"].pct_change()
    px["price_basis"] = px["adj_close"]                          # continuous, total-return basis
    px["vol_basis"] = px["volume"].astype(float)
    px["era"] = "initial"
    return px[["symbol", "date", "close", "a", "ret", "price_basis", "vol_basis", "era"]]


def load_daily(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] == "20"):
        f = d / "prices.csv"
        if f.exists() and f.stat().st_size > 0:
            frames.append(pd.read_csv(f, usecols=["symbol", "date", "close", "volume"], dtype={"symbol": str}))
    px = pd.concat(frames, ignore_index=True)
    px = px[_clean_symbol_mask(px["symbol"])]
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["volume"] = pd.to_numeric(px["volume"], errors="coerce")
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px[px["date"] > "2015-03-20"]
    px = px[px["close"] > 0.011]          # 0.01 is a placeholder print in this feed
    px = px.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
    return px


def detect_splits(daily: pd.DataFrame, last_initial_close: pd.Series) -> pd.DataFrame:
    """Return daily frame with split-corrected returns and cumulative split factor C (2015 basis)."""
    daily = daily.copy()
    prev = daily.groupby("symbol")["close"].shift(1)
    first = prev.isna()
    prev = prev.where(~first, daily["symbol"].map(last_initial_close))
    raw = daily["close"] / prev
    # a split shows up as a move that is (i) large and (ii) within 4% of a standard ratio;
    # the price ratio of an N:1 split is 1/N
    ratio_to_split = np.abs(raw.to_numpy()[:, None] / SPLIT_RATIOS[None, :] - 1)
    best = ratio_to_split.argmin(axis=1)
    close_enough = ratio_to_split[np.arange(len(best)), best] < 0.045
    big = ((raw < 0.8) | (raw > 1.3)).to_numpy() & raw.notna().to_numpy()
    cand_mult = SPLIT_RATIOS[best]
    # (i) companies split forward when the price is high and reverse-split when it is low
    forward = cand_mult < 1
    price_ok = np.where(forward, prev.to_numpy() >= 20, prev.to_numpy() < 10)
    # (ii) share volume must jump by roughly the split ratio and stay there (a crash spikes volume
    #      only briefly and does not shrink it after a reverse split)
    v = daily["volume"].astype(float).clip(lower=1)
    g = daily.assign(v=v).groupby("symbol")["v"]
    pre = g.transform(lambda x: x.shift(1).rolling(20, min_periods=5).median())
    post = g.transform(lambda x: x[::-1].rolling(20, min_periods=5).median()[::-1].shift(-1))
    vol_ratio = (post / pre).to_numpy()
    expected = np.log(1 / cand_mult)
    vol_ok = np.sign(np.log(vol_ratio)) == np.sign(expected)
    vol_ok &= np.abs(np.log(vol_ratio)) > 0.4 * np.abs(expected)
    # (iii) on the split day itself volume is ~N x normal; a crash day prints far more than that
    day0 = (v / pre).to_numpy()
    n_ratio = np.maximum(1 / cand_mult, cand_mult)
    day0_ok = np.where(forward, day0 <= 3.0 * n_ratio, True)
    normal = big & close_enough & price_ok & day0_ok & np.nan_to_num(vol_ok, nan=False).astype(bool)
    # (iv) large reverse splits (1:5 and beyond) on sub-$10 stocks: the split day is often volatile and
    #      volume does not shrink cleanly, but a genuine 5x+ one-day jump is far rarer than a reverse split
    dev = ratio_to_split[np.arange(len(best)), best]
    big_reverse = (cand_mult >= 5) & (dev < 0.3) & (prev.to_numpy() < 10) & big
    is_split = normal | big_reverse
    split_factor = np.where(is_split, cand_mult, 1.0)
    daily["split_mult"] = split_factor
    daily["ret"] = raw / daily["split_mult"] - 1
    # new listings often print garbage in their first days, and a long gap usually means a
    # recycled ticker: do not chain returns across either
    gap = daily.groupby("symbol")["date"].diff().dt.days
    new_listing = ~daily["symbol"].isin(last_initial_close.index)
    first_days = daily.groupby("symbol").cumcount() < 10
    daily.loc[(new_listing & first_days) | (gap > 10), "ret"] = np.nan
    daily = _drop_spikes(daily)
    # C_t: multiply raw price by C to return to 2015 basis -> C = 1 / prod(split_mult)
    daily["C"] = 1.0 / daily.groupby("symbol")["split_mult"].cumprod()
    daily["price_basis"] = daily["close"] * daily["C"]
    daily["vol_basis"] = daily["volume"].astype(float) / daily["C"]
    daily["era"] = "live"
    return daily


def _drop_spikes(daily: pd.DataFrame, jump: float = 0.8, window: int = 10, back_tol: float = 0.25) -> pd.DataFrame:
    """Blank out bad prints: a >80% move that reverts to within 25% of the old level inside 10 days.

    The live feed occasionally serves a different security under the same ticker for a few days
    (e.g. a $25 unit trading as NEE). Chaining those prices would create fake +/-80% months.
    """
    daily = daily.reset_index(drop=True)
    lvl = np.log((1 + daily["ret"].fillna(0)).groupby(daily["symbol"]).cumprod())
    big = daily.index[np.abs(np.log1p(daily["ret"].fillna(0))) > np.log1p(jump)]
    sym = daily["symbol"].to_numpy()
    lv = lvl.to_numpy()
    kill = np.zeros(len(daily), dtype=bool)
    for i in big:
        if i == 0 or sym[i - 1] != sym[i]:
            continue
        base = lv[i - 1]
        for k in range(i + 1, min(i + 1 + window, len(daily))):
            if sym[k] != sym[i]:
                break
            if abs(lv[k] - base) < np.log1p(back_tol):
                kill[i:k + 1] = True
                break
    daily.loc[kill, "ret"] = np.nan
    return daily


def load_reports(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "reports.csv"
        if f.exists() and f.stat().st_size > 0:
            # daily archives ship reports.csv without a header row; read all files positionally
            df = pd.read_csv(f, header=None, names=REPORT_COLS, dtype=str)
            frames.append(df[df["symbol"] != "symbol"])
    r = pd.concat(frames, ignore_index=True)
    r = r[r["doc_type"].isin(["10-Q", "10-K"])].copy()
    for c in ["net_income", "eps_basic", "eps_diluted"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r["end_date"] = pd.to_datetime(r["end_date"], errors="coerce")
    eps = r["eps_basic"].where(r["eps_basic"].abs() >= 0.02, r["eps_diluted"])
    ok = (eps.abs() >= 0.02) & (r["net_income"].abs() >= 1e5)
    r["shares_raw"] = (r["net_income"] / eps).where(ok)
    r = r[(r["shares_raw"] > 1e5) & (r["shares_raw"] < 5e10)]
    r = r.dropna(subset=["end_date"]).drop_duplicates(["symbol", "end_date"], keep="last")
    r["available"] = r["end_date"] + pd.Timedelta(days=REPORT_LAG_DAYS)
    return r[["symbol", "end_date", "available", "shares_raw"]].sort_values(["symbol", "end_date"])


def main(src: str, out: str) -> None:
    root = Path(src)
    print("loading 2009-2015 batch ...")
    ini = load_initial(root)
    last_close = ini.sort_values("date").groupby("symbol")["close"].last()
    last_basis = ini.sort_values("date").groupby("symbol")["price_basis"].last()

    print("loading 2015-2017 daily snapshots ...")
    live = detect_splits(load_daily(root), last_close)
    # stitch: post-2015 basis price continues the adjusted series for survivors of the initial batch
    scale = live["symbol"].map(last_basis / last_close).fillna(1.0)
    live["price_basis"] = live["price_basis"] * scale
    print(f"  detected {(live['split_mult'] != 1).sum()} splits / reverse splits in the live era")

    daily = pd.concat([ini.drop(columns=["close", "a"]).assign(a=ini["a"], C=np.nan, close=ini["close"]),
                       live[["symbol", "date", "close", "ret", "price_basis", "vol_basis", "era", "C"]]],
                      ignore_index=True).sort_values(["symbol", "date"])
    daily = daily[daily.groupby("symbol")["date"].transform("size") >= MIN_PRICE_ROWS]
    # guard against bad prints: one-day moves beyond -90% / +900% are data errors in this feed
    # (mostly mis-adjusted reverse splits), and spike-and-revert blocks are bad prints
    daily.loc[(daily["ret"] < -0.9) | (daily["ret"] > 9), "ret"] = np.nan
    # a 3x one-day jump in a stock already above $5 is a ticker collision in this feed, not a return
    prev_close = daily.sort_values(["symbol", "date"]).groupby("symbol")["close"].shift(1)
    daily.loc[(daily["ret"] > 2) & (prev_close >= 5), "ret"] = np.nan
    daily = _drop_spikes(daily.sort_values(["symbol", "date"]))

    print("attaching shares outstanding from SEC filings ...")
    rep = load_reports(root)
    # basis factor at each report date: pre-2015 use a (close/adj_close), post-2015 use 1/C
    fac_src = daily[["symbol", "date", "a", "C", "era"]].copy()
    fac_src["basis_fac"] = np.where(fac_src["era"] == "initial", fac_src["a"], 1.0 / fac_src["C"])
    fac_src = fac_src.dropna(subset=["basis_fac"]).sort_values("date")
    rep = rep.sort_values("end_date")
    rep = pd.merge_asof(rep, fac_src[["symbol", "date", "basis_fac"]], left_on="end_date", right_on="date",
                        by="symbol", direction="nearest", tolerance=pd.Timedelta(days=10))
    rep["shares_basis"] = rep["shares_raw"] * rep["basis_fac"]
    rep = rep.dropna(subset=["shares_basis"])[["symbol", "available", "shares_basis"]].sort_values("available")

    daily = daily.sort_values("date")
    back = pd.merge_asof(daily, rep, left_on="date", right_on="available", by="symbol", direction="backward",
                         tolerance=pd.Timedelta(days=400))
    # XBRL coverage of small filers only starts in 2011; for earlier months fall back to the first
    # later filing (share counts drift slowly, and they are only used to rank turnover terciles)
    fwd = pd.merge_asof(daily, rep, left_on="date", right_on="available", by="symbol", direction="forward",
                        tolerance=pd.Timedelta(days=730))
    back["shares_basis"] = back["shares_basis"].fillna(fwd["shares_basis"])
    daily = back
    daily["turnover"] = daily["vol_basis"] / daily["shares_basis"]
    daily["mktcap"] = daily["price_basis"] * daily["shares_basis"]
    bad = (daily["turnover"] > 0.5) | (daily["mktcap"] < 5e6) | (daily["mktcap"] > 5e12)
    daily.loc[bad, ["turnover", "mktcap"]] = np.nan

    print("aggregating to months ...")
    daily["month"] = daily["date"].dt.to_period("M")
    daily["lr"] = np.log1p(daily["ret"])
    g = daily.groupby(["symbol", "month"])
    monthly = pd.DataFrame({
        "ret": np.expm1(g["lr"].sum(min_count=1)),
        "close": g["close"].last(),
        "mktcap": g["mktcap"].last(),
        "turnover": g["turnover"].mean(),           # average daily turnover in the month
        "n_days": g["ret"].count(),
        "era": g["era"].last(),
    }).reset_index()
    # first month of a stock has a partial return; drop it
    monthly = monthly[monthly["n_days"] >= 5]
    monthly["month"] = monthly["month"].dt.to_timestamp("M")
    monthly = monthly.sort_values(["symbol", "month"]).reset_index(drop=True)
    monthly["turnover"] = monthly["turnover"].astype("float32")
    monthly["ret"] = monthly["ret"].astype("float32")
    monthly["mktcap"] = monthly["mktcap"].astype("float32")
    monthly["close"] = monthly["close"].astype("float32")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(out, index=False)
    print(f"wrote {out}: {len(monthly):,} stock-months, {monthly['symbol'].nunique():,} symbols, "
          f"{monthly['month'].min():%Y-%m} .. {monthly['month'].max():%Y-%m}, "
          f"turnover available for {monthly['turnover'].notna().mean():.0%} of rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
