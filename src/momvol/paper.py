"""Published numbers from Lee & Swaminathan (2000), Journal of Finance 55(5), 2017-2069.

Sample: NYSE/AMEX common stocks, January 1965 - December 1995. Mean monthly returns in percent.
Transcribed from Tables I, II and VII so the app can put them next to the modern estimates.
"""
from __future__ import annotations

import pandas as pd

SAMPLE = "NYSE/AMEX, 1965-1995"
DOI = "https://doi.org/10.1111/0022-1082.00280"

# Table I: R10 - R1 monthly return (%) by formation J (rows) and holding K (columns)
TABLE1_R10_R1 = pd.DataFrame(
    [[0.66, 0.67, 0.70, 0.72],
     [1.04, 1.05, 1.08, 0.88],
     [1.36, 1.35, 1.15, 0.88],
     [1.54, 1.26, 1.01, 0.74]],
    index=pd.Index([3, 6, 9, 12], name="J"), columns=pd.Index([3, 6, 9, 12], name="K"))

# Table I, J = 6: formation-period profile
TABLE1_PROFILE_J6 = pd.DataFrame({
    "Portfolio": ["R1", "R5", "R10"],
    "formation return %/month": [-6.36, 0.25, 8.30],
    "avg daily turnover %": [0.1671, 0.1212, 0.2349],
    "median price $": [9.00, 20.79, 19.41],
})

# Table II: {(J, K): {row: [V1, V2, V3, V3-V1]}}
_T2 = {
    3: {"R1": [[1.24, .96, .19, -1.05], [1.19, .87, .25, -.93], [1.21, .89, .34, -.86], [1.17, .81, .36, -.81]],
        "R5": [[1.41, 1.45, 1.20, -.20], [1.42, 1.38, 1.23, -.19], [1.40, 1.34, 1.19, -.21], [1.40, 1.31, 1.14, -.26]],
        "R10": [[1.25, 1.61, 1.45, .20], [1.43, 1.59, 1.36, -.07], [1.54, 1.65, 1.41, -.13], [1.59, 1.65, 1.37, -.23]],
        "R10-R1": [[.01, .66, 1.26, 1.26], [.25, .73, 1.11, .86], [.33, .76, 1.06, .73], [.43, .85, 1.01, .58]]},
    6: {"R1": [[1.16, .77, .03, -1.14], [1.12, .67, .09, -1.04], [1.03, .67, .16, -.88], [1.09, .74, .30, -.79]],
        "R5": [[1.37, 1.34, 1.19, -.18], [1.36, 1.34, 1.15, -.21], [1.38, 1.35, 1.16, -.22], [1.39, 1.32, 1.13, -.26]],
        "R10": [[1.63, 1.82, 1.57, -.06], [1.67, 1.78, 1.55, -.12], [1.72, 1.85, 1.56, -.16], [1.66, 1.75, 1.42, -.23]],
        "R10-R1": [[.47, 1.05, 1.55, 1.07], [.54, 1.11, 1.46, .91], [.69, 1.17, 1.41, .71], [.57, 1.00, 1.13, .56]]},
    9: {"R1": [[1.16, .65, -.14, -1.30], [.99, .54, -.04, -1.02], [1.01, .69, .15, -.86], [1.09, .77, .32, -.77]],
        "R5": [[1.39, 1.33, 1.04, -.35], [1.37, 1.31, 1.09, -.28], [1.40, 1.33, 1.13, -.27], [1.41, 1.31, 1.10, -.31]],
        "R10": [[1.91, 2.09, 1.73, -.17], [1.92, 2.00, 1.67, -.26], [1.86, 1.94, 1.57, -.29], [1.75, 1.79, 1.39, -.35]],
        "R10-R1": [[.74, 1.44, 1.87, 1.13], [.94, 1.46, 1.70, .77], [.85, 1.25, 1.42, .57], [.66, 1.02, 1.07, .41]]},
    12: {"R1": [[.92, .47, -.21, -1.13], [.95, .58, .00, -.94], [1.04, .73, .24, -.80], [1.10, .81, .41, -.69]],
         "R5": [[1.28, 1.33, 1.07, -.21], [1.36, 1.35, 1.10, -.26], [1.40, 1.38, 1.12, -.29], [1.43, 1.34, 1.08, -.35]],
         "R10": [[1.94, 2.09, 1.74, -.20], [1.91, 1.89, 1.57, -.33], [1.82, 1.84, 1.45, -.37], [1.71, 1.67, 1.31, -.40]],
         "R10-R1": [[1.02, 1.62, 1.95, .92], [.96, 1.31, 1.57, .61], [.78, 1.11, 1.21, .43], [.60, .86, .90, .29]]},
}
_KS = [3, 6, 9, 12]


def table2(J: int, K: int) -> pd.DataFrame | None:
    """Paper Table II block for (J, K) with columns V1, V2, V3, V3-V1, or None if not published."""
    if J not in _T2 or K not in _KS:
        return None
    k = _KS.index(K)
    rows = [{"Portfolio": r, "V1": v[k][0], "V2": v[k][1], "V3": v[k][2], "V3-V1": v[k][3]} for r, v in _T2[J].items()]
    return pd.DataFrame(rows)


def r10_r1_by_volume() -> pd.DataFrame:
    """Long table of the paper's R10-R1 spread within each volume tercile, every (J, K)."""
    rows = []
    for J, d in _T2.items():
        for k, K in enumerate(_KS):
            v1, v2, v3, diff = d["R10-R1"][k]
            rows.append({"J": J, "K": K, "R10-R1|V1": v1, "R10-R1|V2": v2, "R10-R1|V3": v3, "V3-V1": diff})
    return pd.DataFrame(rows)


# Table VII, Panel A (raw returns, J = 6): annual event-time returns in % per year
TABLE7 = pd.DataFrame({
    "series": ["R10-R1", "Early: R10V1-R1V3", "Late: R10V3-R1V1"],
    "Year 1": [12.49, 16.70, 6.84],
    "Year 2": [-1.10, 6.19, -5.35],
    "Year 3": [-0.32, 5.85, -3.91],
    "Year 4": [-2.77, 1.53, -6.33],
    "Year 5": [-2.96, -0.11, -4.78],
})
