"""Precompute the slow parts of the study (the full J x K grid, event-time tables, alphas).

    python scripts/run_analysis.py

Writes parquet files to results/. The Streamlit app reads them and recomputes only the
single (J, K) the user selects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momvol import data, portfolios as P, stats as S  # noqa: E402

OUT = ROOT / "results"
FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
SPREADS = ["R10-R1", "R10-R1|V1", "R10-R1|V3", "(R10-R1)V3 - (R10-R1)V1", "Early: R10V1-R1V3",
           "Late: R10V3-R1V1", "V3-V1|R1", "V3-V1|R10", "V1-V3"]
SUBPERIODS = {
    "Full sample": (None, None),
    "2009-2012": ("2009", "2012"),
    "2013-2017": ("2013", "2017"),
    "Excluding 2009": ("2010", None),
    "Live-captured only (Apr 2015+)": ("2015-04", None),
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panel = data.load_panel()
    w = P.to_wide(panel)
    f = data.load_french_factors()

    for skip in (0, 1):
        print(f"grid, skip={skip} ...")
        P.grid_r10_r1(w, skip=skip).to_parquet(OUT / f"grid_skip{skip}.parquet")

    print("event time ...")
    P.event_table(P.event_time(w, J=6, years=2)).to_parquet(OUT / "event_j6.parquet")

    print("alphas and subperiods (J=6, K=6) ...")
    jt = P.add_spreads(P.jt_returns(w, 6, 6))
    rows = []
    for c in SPREADS:
        a5 = S.factor_alpha(jt[c], f, FF5)
        a6 = S.factor_alpha(jt[c], f, FF5 + ["Mom"])
        rows.append({"series": c, "raw_pct": 100 * jt[c].mean(), "raw_t": S.tstat(jt[c]),
                     "ff5_alpha_pct": a5["alpha_pct"], "ff5_t": a5["alpha_t"],
                     "ff5mom_alpha_pct": a6["alpha_pct"], "ff5mom_t": a6["alpha_t"],
                     "b_mkt": a6["b_Mkt-RF"], "b_smb": a6["b_SMB"], "b_hml": a6["b_HML"],
                     "b_rmw": a6["b_RMW"], "b_cma": a6["b_CMA"], "b_mom": a6["b_Mom"]})
    pd.DataFrame(rows).to_parquet(OUT / "alphas_j6k6.parquet")

    rows = []
    for name, (a, b) in SUBPERIODS.items():
        x = jt.loc[a:b]
        for c in SPREADS:
            rows.append({"period": name, "series": c, "months": int(x[c].notna().sum()),
                         "mean_pct": 100 * x[c].mean(), "t": S.tstat(x[c])})
    pd.DataFrame(rows).to_parquet(OUT / "subperiods_j6k6.parquet")

    print("validation against CRSP ...")
    v = P.jt_returns(w, J=11, K=1, skip=1)          # the (12-2) construction Ken French uses
    d = data.load_french_deciles("equal")
    val = pd.DataFrame({"panel_R10-R1": v["R10"] - v["R1"], "crsp_D10-D1": (d["D10"] - d["D1"]).reindex(v.index),
                        "panel_R1": v["R1"], "crsp_D1": d["D1"].reindex(v.index),
                        "panel_R10": v["R10"], "crsp_D10": d["D10"].reindex(v.index)}).dropna()
    val.to_parquet(OUT / "validation.parquet")
    print("done:", sorted(p.name for p in OUT.glob("*.parquet")))


if __name__ == "__main__":
    main()
