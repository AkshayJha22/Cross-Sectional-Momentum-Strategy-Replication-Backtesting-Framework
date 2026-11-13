"""Data loaders: the stock-level monthly panel and Kenneth French's CRSP-based portfolios."""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_panel(path: Path | str = DATA / "panel_monthly.parquet") -> pd.DataFrame:
    """Monthly stock panel built by scripts/build_panel.py (2009-01 .. 2017-03)."""
    p = pd.read_parquet(path)
    p["month"] = pd.to_datetime(p["month"])
    return p


def _french_block(path: Path, header: str) -> pd.DataFrame:
    """Read one monthly table (rows YYYYMM) that follows `header` in a Ken French CSV."""
    lines = path.read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if l.strip().startswith(header))
    start = i if lines[i].lstrip().startswith(",") else i + 1      # the column-name row
    rows = [lines[start]]
    for l in lines[start + 1:]:
        key = l.split(",")[0].strip()
        if len(key) == 6 and key.isdigit():
            rows.append(l)
        elif rows[1:]:
            break
    df = pd.read_csv(StringIO("\n".join(rows)), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan)
    return df / 100.0


def load_french_deciles(weighting: str = "equal") -> pd.DataFrame:
    """10 portfolios formed monthly on prior (t-12..t-2) return. Columns D1 (losers) .. D10 (winners)."""
    header = "Average Equal Weighted Returns -- Monthly" if weighting == "equal" else "Value Weight Returns -- Monthly"
    df = _french_block(DATA / "french" / "10_Portfolios_Prior_12_2.csv", header)
    df.columns = [f"D{i}" for i in range(1, 11)]
    return df


def load_french_factors() -> pd.DataFrame:
    """Fama-French 5 factors + RF, joined with the momentum factor (Mom)."""
    ff = _french_block(DATA / "french" / "F-F_Research_Data_5_Factors_2x3.csv", ",Mkt-RF")
    mom = _french_block(DATA / "french" / "F-F_Momentum_Factor.csv", ",Mom")
    mom.columns = ["Mom"]
    return ff.join(mom, how="outer")


def french_vintage() -> str:
    first = (DATA / "french" / "10_Portfolios_Prior_12_2.csv").read_text().splitlines()[0]
    return first.replace("This file was created using the", "").replace("database.", "").strip()
